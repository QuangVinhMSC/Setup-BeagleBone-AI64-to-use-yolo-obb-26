"""Cut one attention block out of a head-cut ONNX for an isolated TIDL test, with real calibration tensors.

  python scripts/extract_block.py onnx/yolo-26-obb-weight_one2one_320_deit.onnx --block 0 --out onnx/blk_deit
  -> <out>/model.onnx  (input = attention input [1,128,10,10], output = proj output, before the residual Add)
     <out>/calib.npz   (that input on every test image, key "<frame>|<input name>", as compile_tidl.py --feeds wants)
     <out>/input.npy   (frame 0, for op_test.py)

The block is found as: the residual Add right after the block's MatMuls, Add(x, attention(x)).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obb_common import letterbox  # noqa: E402


def find_block(m, block):
    nodes = list(m.graph.node)
    producer = {o: i for i, n in enumerate(nodes) for o in n.output}
    mm = [i for i, n in enumerate(nodes) if n.op_type == "MatMul"]
    # group MatMuls into blocks: a gap of more than 30 nodes starts a new block
    groups = [[mm[0]]]
    for i in mm[1:]:
        (groups[-1].append(i) if i - groups[-1][-1] < 30 else groups.append([i]))
    last = groups[block][-1]
    for n in nodes[last:]:
        if n.op_type == "Add":
            a, b = n.input
            if producer.get(a, -1) < groups[block][0] < producer.get(b, 1 << 30):
                return a, b  # (attention input, attention output)
    raise SystemExit("residual Add not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = onnx.load(a.onnx)
    t_in, t_out = find_block(m, a.block)
    os.makedirs(a.out, exist_ok=True)
    model = os.path.join(a.out, "model.onnx")
    onnx.utils.extract_model(a.onnx, model, [t_in], [t_out])
    blk = onnx.load(model)
    print("block %d: %s -> %s, %d nodes, ops %s" % (a.block, t_in, t_out, len(blk.graph.node),
                                                  sorted({n.op_type for n in blk.graph.node})))

    # real block inputs: run the full model with t_in exposed as an extra output
    m.graph.output.extend([onnx.ValueInfoProto(name=t_in)])
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    xs = [sess.run([t_in], {"images": letterbox(cv2.imread(f), 320)[0]})[0]
          for f in sorted(glob.glob("test-images/*.png"))]
    np.savez(os.path.join(a.out, "calib.npz"), **{"%d|%s" % (k, t_in): x for k, x in enumerate(xs)})
    np.save(os.path.join(a.out, "input.npy"), xs[0])
    print("saved", a.out, "input range %.2f..%.2f" % (min(x.min() for x in xs), max(x.max() for x in xs)))


if __name__ == "__main__":
    main()
