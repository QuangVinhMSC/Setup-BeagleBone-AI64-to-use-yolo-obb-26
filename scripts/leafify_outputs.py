"""Make every TIDL-subgraph output a leaf tensor (workaround for a TIDL 8.2 target hang).

TIDL 8.2 (add_data_convert_ops=3) hangs the C7x while initialising the output DataConvert layer of a subgraph
output that is ALSO consumed inside the same subgraph (e.g. backbone P3/P4 that feed both the next backbone conv and,
across the ARM-side attention, the neck Concat). PC emulation is fine; only the target hangs.

Fix: predict the TIDL subgraphs (connected components of non-denied nodes), and for every tensor that is used both
inside its component and outside it, route the outside consumers through an identity 3x3 depthwise conv. The copy
lands in the producing subgraph and becomes a leaf output; the original tensor stays internal.

  python scripts/leafify_outputs.py onnx/yolo-26-obb-weight_one2one_320.onnx      # rewrites in place (+ .json kept)
"""
import argparse
import json
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

DENY = "MatMul,Softmax,Transpose,Reshape,Split"  # keep in sync with compile_tidl.py --deny


def components(g, deny):
    """Predict TIDL subgraphs -> {node_index: component_id} for non-denied nodes.

    A node that (transitively) depends on an ARM node cannot share a subgraph with nodes upstream of that ARM node
    (it would form a cycle), so each node gets a level = number of TIDL->ARM->TIDL crossings above it, and only
    same-level TIDL nodes joined by a tensor edge are merged. Nodes are assumed topologically sorted (ONNX requires it).
    """
    tidl = [n.op_type not in deny for n in g.node]
    prod = {o: i for i, n in enumerate(g.node) for o in n.output}
    level = [0] * len(g.node)
    for i, n in enumerate(g.node):
        for t in n.input:
            j = prod.get(t)
            if j is not None:
                level[i] = max(level[i], level[j] + (1 if tidl[i] and not tidl[j] else 0))

    parent = list(range(len(g.node)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, n in enumerate(g.node):
        if not tidl[i]:
            continue
        for t in n.input:
            j = prod.get(t)
            if j is not None and tidl[j] and level[j] == level[i]:
                parent[find(i)] = find(j)
    return {i: find(i) for i in range(len(g.node)) if tidl[i]}, prod, level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--deny", default=DENY)
    a = ap.parse_args()
    out = a.out or a.onnx
    deny = set(a.deny.split(","))

    m = onnx.load(a.onnx)
    g = m.graph
    comp, prod, _ = components(g, deny)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(shape_inference.infer_shapes(m).graph.value_info) + list(g.output)}
    graph_outs = {o.name for o in g.output}

    # TIDL's partitioner is greedy in node order and sensitive to where the copies go: right after the producer or
    # right before the outside consumer both changed the split. Insert each copy at the end of the producer's
    # contiguous TIDL run (before the next ARM node), or before its first outside consumer if that comes sooner.
    arm = [j for j, c in enumerate(g.node) if c.op_type in deny]
    before, n_fix = {}, 0
    for i, n in enumerate(g.node):
        if i not in comp:
            continue
        for t in n.output:
            cons = [j for j, c in enumerate(g.node) if t in c.input]
            inside = [j for j in cons if comp.get(j) == comp[i]]
            outside = [j for j in cons if comp.get(j) != comp[i]]
            if not inside or not (outside or t in graph_outs):
                continue
            if not outside:  # graph output that is also used internally: leave (head outputs are leaves anyway)
                continue
            c = shapes[t][1]
            w = np.zeros((c, 1, 3, 3), np.float32)
            w[:, 0, 1, 1] = 1.0
            wn, bn, tn = "lw%d" % n_fix, "lb%d" % n_fix, "%s_leaf" % t
            g.initializer.extend([numpy_helper.from_array(w, wn), numpy_helper.from_array(np.zeros(c, np.float32), bn)])
            pos = min([min(outside)] + [j for j in arm if j > i])
            before.setdefault(pos, []).append(
                helper.make_node("Conv", [t, wn, bn], [tn], name="nleaf%d" % n_fix, group=c,
                                 kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
            for j in outside:
                g.node[j].input[:] = [tn if x == t else x for x in g.node[j].input]
            print("leaf copy for %s %s: %d inside, %d outside consumers" % (t, shapes[t], len(inside), len(outside)))
            n_fix += 1
    new_nodes = []
    for j, n in enumerate(g.node):
        new_nodes.extend(before.get(j, []))
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    m = shape_inference.infer_shapes(m)  # TIDL import needs a static shape on every tensor, incl. the new copies
    onnx.checker.check_model(m)
    onnx.save(m, out)
    if out != a.onnx:
        shutil.copy(a.onnx[:-5] + ".json", out[:-5] + ".json")
    meta = json.load(open(out[:-5] + ".json"))
    meta["leaf_copies"] = n_fix
    json.dump(meta, open(out[:-5] + ".json", "w"), indent=1)
    print("inserted %d leaf copies -> %s" % (n_fix, out))


if __name__ == "__main__":
    main()
