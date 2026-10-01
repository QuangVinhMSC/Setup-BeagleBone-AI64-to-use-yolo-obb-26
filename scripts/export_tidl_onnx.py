"""Export an Ultralytics YOLO26-OBB checkpoint to a TIDL-friendly ONNX.

What is different from `yolo export format=onnx`:
  * Conv+BN fused, and the /255 input normalisation folded into the first conv weights,
    so the model takes raw 0..255 pixels (no extra op for TIDL).
  * The graph is cut right after the head convolutions. It returns 9 raw tensors
    (box, cls, angle) x (stride 8, 16, 32). Box decoding, sigmoid, top-k / NMS run on the
    ARM in numpy (see obb_postprocess.py). TIDL 8.2 cannot run TopK/GatherElements/Sin/Cos,
    and keeping each output separate gives each its own quantisation scale.
  * SPPF MaxPool 5x5/s1 (unsupported by TIDL 8.2) replaced by two cascaded 3x3/s1 pools (identical result).
  * Fixed input 1x3xSxS, opset 11 (TIDL 8.2 / onnxruntime 1.7 on the board).

Usage:
  python export_tidl_onnx.py yolo-26-obb-weight.pt     --branch one2many --imgsz 320
  python export_tidl_onnx.py yolo-26-obb-weight_e2e.pt --branch one2one  --imgsz 320
"""
import argparse
import json
import os

import onnx
import torch
import torch.nn as nn


class HeadCut(nn.Module):
    """Backbone + neck + one head branch, returning raw per-level conv outputs."""

    def __init__(self, det_model, branch):
        super().__init__()
        self.layers = det_model.model[:-1]
        self.save = det_model.save
        head = det_model.model[-1]
        self.head_from = head.f
        pre = "one2one_" if branch == "one2one" else ""
        self.box = getattr(head, pre + "cv2")
        self.cls = getattr(head, pre + "cv3")
        self.ang = getattr(head, pre + "cv4")
        self.nl = head.nl

    def forward(self, x):
        y = []
        for m in self.layers:  # same routing as DetectionModel._predict_once
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        feats = [y[j] for j in self.head_from]
        outs = []
        for i in range(self.nl):
            outs += [self.box[i](feats[i]), self.cls[i](feats[i]), self.ang[i](feats[i])]
        return tuple(outs)


def fold_attention_scale(att):
    """Multiply the q rows of the (already fused) qkv conv by att.scale and drop the runtime Mul.
    qkv channels are laid out per head as [q(key_dim), k(key_dim), v(head_dim)]."""
    per_head = att.key_dim * 2 + att.head_dim
    conv = att.qkv.conv
    with torch.no_grad():
        for h in range(att.num_heads):
            rows = slice(h * per_head, h * per_head + att.key_dim)
            conv.weight[rows] *= att.scale
            conv.bias[rows] *= att.scale

    def forward(x, self=att):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, per_head, H * W).split([self.key_dim, self.key_dim, self.head_dim], 2)
        attn = (q.transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)

    att.forward = forward


def _sub_conv(conv, rows):
    """New conv holding only the given output rows (and, for a depthwise conv, the matching groups)."""
    rows = torch.as_tensor(rows)
    dw = conv.groups > 1
    c = nn.Conv2d(len(rows) if dw else conv.in_channels, len(rows), conv.kernel_size, conv.stride, conv.padding,
                  groups=len(rows) if dw else 1, bias=True)
    with torch.no_grad():
        c.weight.copy_(conv.weight[rows])
        c.bias.copy_(conv.bias[rows])
    return c


def deit_attention(att, unroll=False):
    """Re-express the (scale-folded) attention in TI's validated DeiT form, same math:
    q @ kT -> softmax(width) -> attn @ v, no Split (separate q/k/v convs), no transpose of the NxN map,
    pe fed straight from the v conv. unroll=True also unrolls the heads (3-D MatMuls, per-head convs, Concat)."""
    nh, kd, hd = att.num_heads, att.key_dim, att.head_dim
    per_head = 2 * kd + hd
    qkv = att.qkv.conv  # rows per head: [q(kd), k(kd), v(hd)]; scale already folded into q
    rq = [[h * per_head + i for i in range(kd)] for h in range(nh)]
    rk = [[h * per_head + kd + i for i in range(kd)] for h in range(nh)]
    rv = [[h * per_head + 2 * kd + i for i in range(hd)] for h in range(nh)]
    pe = att.pe.conv  # depthwise over v channels, ordered [h0 v.., h1 v..]
    if unroll:
        att.q_h = nn.ModuleList(_sub_conv(qkv, r) for r in rq)
        att.k_h = nn.ModuleList(_sub_conv(qkv, r) for r in rk)
        att.v_h = nn.ModuleList(_sub_conv(qkv, r) for r in rv)
        att.pe_h = nn.ModuleList(_sub_conv(pe, range(h * hd, (h + 1) * hd)) for h in range(nh))
    else:
        flat = lambda rr: [i for r in rr for i in r]  # noqa: E731
        att.q_c, att.k_c, att.v_c = _sub_conv(qkv, flat(rq)), _sub_conv(qkv, flat(rk)), _sub_conv(qkv, flat(rv))
    del att.qkv

    def forward(x, self=att):
        B, C, H, W = x.shape
        N = H * W
        if unroll:
            outs = []
            for h in range(nh):
                q = self.q_h[h](x).view(B, kd, N).transpose(1, 2)  # [B,N,kd]
                k = self.k_h[h](x).view(B, kd, N)  # [B,kd,N] = kT
                vf = self.v_h[h](x)  # [B,hd,H,W]
                attn = (q @ k).softmax(dim=-1)  # [B,N,N]
                o = (attn @ vf.view(B, hd, N).transpose(1, 2)).transpose(1, 2).reshape(B, hd, H, W)
                outs.append(o + self.pe_h[h](vf))
            return self.proj(torch.cat(outs, 1))
        q = self.q_c(x).view(B, nh, kd, N).transpose(-2, -1)  # [B,nh,N,kd]
        k = self.k_c(x).view(B, nh, kd, N)  # [B,nh,kd,N]
        vf = self.v_c(x)  # [B,C,H,W]
        attn = (q @ k).softmax(dim=-1)  # [B,nh,N,N]
        o = (attn @ vf.view(B, nh, hd, N).transpose(-2, -1)).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(o + self.pe(vf))

    att.forward = forward


def shorten_names(m):
    """TIDL 8.2's importer has fixed-size name buffers (crashes with 'buffer overflow detected').
    Rename internal tensors to t<i> and nodes to n<i>; graph input/output names are kept."""
    keep = {i.name for i in m.graph.input} | {o.name for o in m.graph.output} | {t.name for t in m.graph.initializer}
    ren = {}
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}"
        for j, o in enumerate(n.output):
            if o not in keep:
                ren[o] = ren.get(o, f"t{i}_{j}")
                n.output[j] = ren[o]
    for n in m.graph.node:
        for j, x in enumerate(n.input):
            n.input[j] = ren.get(x, x)
    for v in m.graph.value_info:
        v.name = ren.get(v.name, v.name)
    for k, t in enumerate(m.graph.initializer):  # initializers too
        new = f"w{k}"
        for n in m.graph.node:
            for j, x in enumerate(n.input):
                if x == t.name:
                    n.input[j] = new
        t.name = new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--branch", choices=["one2many", "one2one"], required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", default=None)
    # orig = Ultralytics layout (Split, qT@k, v@attnT); deit / deit_unroll = TI's validated transformer pattern (TIDL 10)
    ap.add_argument("--attn", choices=["orig", "deit", "deit_unroll"], default="orig")
    ap.add_argument("--opset", type=int, default=11)
    a = ap.parse_args()

    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck["model"]).float().eval()
    head = model.model[-1]
    strides = [int(s) for s in head.stride.tolist()]
    names = {int(k): v for k, v in model.names.items()}

    cut = HeadCut(model, a.branch)
    for m in cut.modules():  # fuse Conv+BN (same as Ultralytics Conv.fuse path)
        if hasattr(m, "bn") and hasattr(m, "conv") and hasattr(m, "forward_fuse"):
            from ultralytics.utils.torch_utils import fuse_conv_and_bn

            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, "bn")
            m.forward = m.forward_fuse
    for mod in list(cut.modules()):  # TIDL 8.2: MaxPool k5/s1 unsupported -> two k3/s1 pools (exactly equivalent)
        for n, c in mod.named_children():
            if isinstance(c, nn.MaxPool2d) and c.kernel_size in (5, (5, 5)) and c.stride in (1, (1, 1)):
                setattr(mod, n, nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.MaxPool2d(3, 1, 1)))
    for mod in cut.modules():  # fold attention's q*scale into the qkv conv -> no stray Mul inside the ARM-only block
        if type(mod).__name__ == "Attention":
            fold_attention_scale(mod)
            if a.attn != "orig":
                deit_attention(mod, unroll=a.attn == "deit_unroll")
    first = cut.layers[0].conv
    with torch.no_grad():  # fold x/255 into the first conv
        first.weight.mul_(1.0 / 255.0)

    base = os.path.splitext(os.path.basename(a.weights))[0]
    tag = "" if a.attn == "orig" else "_" + a.attn
    tag += "" if a.opset == 11 else f"_op{a.opset}"
    out = a.out or f"onnx/{base}_{a.branch}_{a.imgsz}{tag}.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out_names = [f"{t}_s{s}" for s in strides for t in ("box", "cls", "ang")]
    dummy = torch.zeros(1, 3, a.imgsz, a.imgsz)
    torch.onnx.export(cut, dummy, out, opset_version=a.opset, input_names=["images"], output_names=out_names,
                      do_constant_folding=True, dynamo=False)

    m = onnx.load(out)
    try:
        import onnxsim

        m, ok = onnxsim.simplify(m)
        assert ok
    except ImportError:
        pass
    shorten_names(m)
    m.ir_version = min(m.ir_version, 7)  # onnxruntime 1.7 on the board reads IR <= 7
    onnx.save(m, out)

    meta = {"weights": a.weights, "branch": a.branch, "imgsz": a.imgsz, "strides": strides, "names": names,
            "outputs": out_names, "input": "images", "input_range": "0..255 RGB, NCHW float32, letterbox pad 114"}
    json.dump(meta, open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"saved {out}\n ops: {ops}\n outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")


if __name__ == "__main__":
    main()
