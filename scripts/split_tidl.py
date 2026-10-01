"""Split a head-cut ONNX into explicit TIDL / CPU parts (run on the PC).

Why: letting TIDL 8.2 partition the graph itself (deny_list) gives subgraphs whose outputs are also consumed inside
the same subgraph (backbone P3/P4). On the target the C7x hangs while initialising the output DataConvert for such
tensors. Here the split is explicit, every TIDL part is compiled as a whole standalone model (no deny list, no greedy
partitioning), and every part output is made a leaf with an identity depthwise conv copy.

  python scripts/split_tidl.py onnx/yolo-26-obb-weight_one2one_320.onnx
  -> onnx/yolo-26-obb-weight_one2one_320_split/{pipeline.json, model.json, s00_tidl.onnx, s01_cpu.onnx, ...}

Stage assignment: nodes whose op type is in --cpu-ops run on the ARM; the remaining nodes form TIDL islands
(connected, same "level" = no path through a CPU node in between, so the stage graph is acyclic). Islands with fewer
than --min-tidl nodes are demoted to CPU (e.g. the lone qkv conv inside an attention block).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

CPU_OPS = "MatMul,Softmax,Transpose,Reshape,Split"
SINK_OPS = ("Reshape", "Transpose")


def assign_stages(g, is_tidl):
    """Returns per-node (level, stage_root). Kind changes along an edge bump the level; same-kind same-level
    neighbours are merged. Stages sorted by level form a valid execution order."""
    prod = {o: i for i, n in enumerate(g.node) for o in n.output}
    level = [0] * len(g.node)
    for i, n in enumerate(g.node):
        for t in n.input:
            j = prod.get(t)
            if j is not None:
                level[i] = max(level[i], level[j] + (1 if is_tidl[i] != is_tidl[j] else 0))
    # TIDL 10 import fails (tidl_optimizeNet) when a part ends on a Reshape/Transpose output, e.g. the transposed v
    # of an attention block that is only used after the CPU Softmax. Sink such pure data-movement nodes into the
    # (later) TIDL stage that consumes them.
    cons = {}
    for i, n in enumerate(g.node):
        for t in n.input:
            if t in prod:
                cons.setdefault(prod[t], []).append(i)
    for i in reversed(range(len(g.node))):
        c = cons.get(i, [])
        if g.node[i].op_type in SINK_OPS and is_tidl[i] and c and all(is_tidl[j] for j in c):
            level[i] = max(level[i], min(level[j] for j in c))
    parent = list(range(len(g.node)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, n in enumerate(g.node):
        for t in n.input:
            j = prod.get(t)
            if j is not None and is_tidl[i] == is_tidl[j] and level[i] == level[j]:
                parent[find(i)] = find(j)
    return level, [find(i) for i in range(len(g.node))]


def split(g, cpu_ops, min_tidl):
    is_tidl = [n.op_type not in cpu_ops for n in g.node]
    for _ in range(5):  # demoting a small island can merge CPU groups, so iterate to a fixed point
        level, root = assign_stages(g, is_tidl)
        sizes = {}
        for i, r in enumerate(root):
            if is_tidl[i]:
                sizes[r] = sizes.get(r, 0) + 1
        small = {r for r, s in sizes.items() if s < min_tidl}
        if not small:
            break
        is_tidl = [t and root[i] not in small for i, t in enumerate(is_tidl)]
    stages = {}
    for i, r in enumerate(root):
        stages.setdefault(r, []).append(i)
    order = sorted(stages, key=lambda r: (level[r], min(stages[r])))
    return [(is_tidl[stages[r][0]], stages[r]) for r in order]


def vi(name, shapes):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name])


def build_part(m, g, idx, kind_tidl, shapes, inits, graph_in, graph_out, consumers):
    nodes = [g.node[i] for i in idx]
    inside = set(idx)
    produced = {o for n in nodes for o in n.output}
    ins, outs = [], []
    for n in nodes:
        for t in n.input:
            if t and t not in produced and t not in inits and t not in ins:
                ins.append(t)
    for n in nodes:
        for t in n.output:
            ext = [j for j in consumers.get(t, []) if j not in inside]
            if ext or t in graph_out:
                outs.append(t)
    new_nodes = list(nodes)
    out_map = []  # (name inside part, global tensor name)
    initializers = [inits[t] for n in nodes for t in n.input if t in inits]
    for k, t in enumerate(outs):
        used_inside = any(j in inside for j in consumers.get(t, []))
        # TIDL 10's importer also drops/rejects a part output that comes straight from a Split/Reshape/Transpose
        movement = any(n.op_type in ("Split",) + SINK_OPS and t in n.output for n in nodes)
        if kind_tidl and (used_inside or movement):
            c = shapes[t][1]
            w = np.zeros((c, 1, 3, 3), np.float32)
            w[:, 0, 1, 1] = 1.0
            wn, bn, tn = "lw%d_%d" % (idx[0], k), "lb%d_%d" % (idx[0], k), t + "_leaf"
            initializers += [numpy_helper.from_array(w, wn), numpy_helper.from_array(np.zeros(c, np.float32), bn)]
            new_nodes.append(helper.make_node("Conv", [t, wn, bn], [tn], name="nleaf%d_%d" % (idx[0], k), group=c,
                                              kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
            shapes[tn] = shapes[t]
            out_map.append((tn, t))
        else:
            out_map.append((t, t))
    seen, uniq = set(), []
    for x in initializers:
        if x.name not in seen:
            seen.add(x.name)
            uniq.append(x)
    graph = helper.make_graph(new_nodes, "part", [vi(t, shapes) for t in ins], [vi(p, shapes) for p, _ in out_map],
                              initializer=uniq)
    pm = helper.make_model(graph, opset_imports=m.opset_import, producer_name="split_tidl")
    pm.ir_version = m.ir_version
    pm = shape_inference.infer_shapes(pm)
    onnx.checker.check_model(pm)
    return pm, ins, out_map


def uint8_input(pm, name):
    """Make the image input uint8 + Cast, like TI's model zoo (TIDL 8.2 hangs on the target with a float32 image
    input; the letterboxed pixels are integers 0..255 anyway, so this is lossless)."""
    g = pm.graph
    cast = name + "_f"
    for n in g.node:
        n.input[:] = [cast if x == name else x for x in n.input]
    g.node.insert(0, helper.make_node("Cast", [name], [cast], name="ncast_in", to=TensorProto.FLOAT))
    for i in g.input:
        if i.name == name:
            i.type.tensor_type.elem_type = TensorProto.UINT8
    onnx.checker.check_model(pm)
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--cpu-ops", default=CPU_OPS)
    ap.add_argument("--min-tidl", type=int, default=8, help="TIDL islands smaller than this run on the ARM")
    ap.add_argument("--float-input", action="store_true", help="keep a float32 image input (hangs on target)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.onnx[:-5] + "_split"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    m = shape_inference.infer_shapes(onnx.load(a.onnx))
    g = m.graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(g.value_info) + list(g.input) + list(g.output)}
    inits = {x.name: x for x in g.initializer}
    graph_in = [x.name for x in g.input if x.name not in inits]
    graph_out = [x.name for x in g.output]
    consumers = {}
    for j, n in enumerate(g.node):
        for t in n.input:
            consumers.setdefault(t, []).append(j)

    stages = []
    for k, (kind_tidl, idx) in enumerate(split(g, set(a.cpu_ops.split(",")), a.min_tidl)):
        name = "s%02d_%s" % (k, "tidl" if kind_tidl else "cpu")
        pm, ins, out_map = build_part(m, g, idx, kind_tidl, shapes, inits, graph_in, graph_out, consumers)
        if graph_in[0] in ins and not a.float_input:
            pm = uint8_input(pm, graph_in[0])
        onnx.save(pm, os.path.join(out, name + ".onnx"))
        ops = sorted({g.node[i].op_type for i in idx})
        stages.append({"name": name, "kind": "tidl" if kind_tidl else "cpu", "model": name + ".onnx",
                       "inputs": ins, "outputs": [[p, t] for p, t in out_map]})
        print("%s: %3d nodes, in %s -> out %s%s" % (name, len(idx), ins, [t for _, t in out_map],
                                                    "" if kind_tidl else "  ops " + ",".join(ops)))

    meta = json.load(open(a.onnx[:-5] + ".json"))
    json.dump(meta, open(os.path.join(out, "model.json"), "w"), indent=1)
    json.dump({"input": graph_in[0], "input_dtype": "float32" if a.float_input else "uint8",
               "outputs": graph_out, "stages": stages},
              open(os.path.join(out, "pipeline.json"), "w"), indent=1)
    print("wrote", out)

    # check: pipeline on CPU == original model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import onnxruntime as ort
    from pipeline import Pipeline
    x = np.random.RandomState(0).randint(0, 256, (1, 3, meta["imgsz"], meta["imgsz"])).astype(np.float32)  # integer pixels
    ref = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"]).run(None, {graph_in[0]: x})
    got = Pipeline(out).run(x)
    print("pipeline vs original, max abs diff: %.2e" % max(float(np.abs(r - o).max()) for r, o in zip(ref, got)))


if __name__ == "__main__":
    main()
