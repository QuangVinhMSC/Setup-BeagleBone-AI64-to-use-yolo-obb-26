"""Tiny ONNX models to probe which attention ops the TDA4VM C7x accepts on TIDL 10 (op_test.py on the board).

  python scripts/make_micro.py            -> onnx/micro/<name>/{model.onnx, calib.npz, input.npy}
Each starts from a float [1,64,10,10] input and a 1x1 conv, like the attention block does.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

rng = np.random.RandomState(0)
N = 100


def conv(name, x, cin, cout):
    w = nh.from_array((rng.randn(cout, cin, 1, 1) * (1.0 / np.sqrt(cin))).astype(np.float32), name + "_w")
    b = nh.from_array(np.zeros(cout, np.float32), name + "_b")
    return [h.make_node("Conv", [x, name + "_w", name + "_b"], [name], kernel_shape=[1, 1])], [w, b]


def reshape(name, x, shape):
    s = nh.from_array(np.array(shape, np.int64), name + "_s")
    return [h.make_node("Reshape", [x, name + "_s"], [name])], [s]


def build(kind):
    nodes, inits = [], []

    def add(r):
        nodes.extend(r[0]); inits.extend(r[1])

    add(conv("c0", "x", 64, 64))
    if kind == "t4":  # 4-D transpose of the last two dims (q^T in deit form)
        add(reshape("r", "c0", [1, 2, 32, N])); nodes.append(h.make_node("Transpose", ["r"], ["y"], perm=[0, 1, 3, 2]))
    elif kind == "t3":  # 3-D transpose, no head dim
        add(reshape("r", "c0", [1, 64, N])); nodes.append(h.make_node("Transpose", ["r"], ["y"], perm=[0, 2, 1]))
    elif kind == "nhwc":  # NCHW -> NHWC
        nodes.append(h.make_node("Transpose", ["c0"], ["y"], perm=[0, 2, 3, 1]))
    elif kind == "mm":  # plain MatMul, no transpose anywhere ([2,100,32] @ [2,32,100])
        add(conv("c1", "x", 64, 64))
        add(reshape("a", "c0", [1, 2, N, 32])); add(reshape("b", "c1", [1, 2, 32, N]))
        nodes.append(h.make_node("MatMul", ["a", "b"], ["y"]))
    elif kind == "mm3":  # same with 3-D tensors (one head)
        add(conv("c1", "x", 64, 32)); add(conv("c2", "x", 64, 32))
        add(reshape("a", "c1", [1, N, 32])); add(reshape("b", "c2", [1, 32, N]))
        nodes.append(h.make_node("MatMul", ["a", "b"], ["y"]))
    elif kind == "sm":  # softmax over the width axis of a [1,2,100,100] map
        add(conv("c1", "x", 64, 200)); add(reshape("a", "c1", [1, 2, N, N]))
        nodes.append(h.make_node("Softmax", ["a"], ["y"], axis=3))
    elif kind == "sm2":  # softmax (width axis) on a reshaped conv output, no MatMul in front
        add(reshape("a", "c0", [1, 2, 32, N])); nodes.append(h.make_node("Softmax", ["a"], ["y"], axis=3))
    elif kind == "sm4":  # softmax over the W axis of a plain NCHW conv output
        nodes.append(h.make_node("Softmax", ["c0"], ["y"], axis=3))
    elif kind == "sm1":  # softmax over all 100 tokens of a [1,64,100] tensor (3-D, last axis)
        add(reshape("a", "c0", [1, 64, N])); nodes.append(h.make_node("Softmax", ["a"], ["y"], axis=2))
    elif kind == "mmsm3":  # MatMul + softmax with 3-D tensors via 4-D reshape of a 64-ch conv: [1,100,32]@[1,32,100]
        add(conv("c1", "x", 64, 64))
        add(reshape("a", "c0", [1, 2 * N, 32])); add(reshape("b", "c1", [1, 32, 2 * N]))
        nodes.append(h.make_node("MatMul", ["a", "b"], ["s"])); nodes.append(h.make_node("Softmax", ["s"], ["y"], axis=2))
    elif kind == "mmsm":  # plain MatMul + softmax (no transpose)
        add(conv("c1", "x", 64, 64))
        add(reshape("a", "c0", [1, 2, N, 32])); add(reshape("b", "c1", [1, 2, 32, N]))
        nodes.append(h.make_node("MatMul", ["a", "b"], ["s"])); nodes.append(h.make_node("Softmax", ["s"], ["y"], axis=3))
    elif kind == "qk":  # deit q^T @ k: Transpose feeding MatMul A
        add(conv("c1", "x", 64, 64))
        add(reshape("a", "c0", [1, 2, 32, N])); add(reshape("b", "c1", [1, 2, 32, N]))
        nodes.append(h.make_node("Transpose", ["a"], ["at"], perm=[0, 1, 3, 2]))
        nodes.append(h.make_node("MatMul", ["at", "b"], ["y"]))
    elif kind == "vaT":  # v @ attn^T: Transpose feeding MatMul B (orig form; also what TIDL fused deit's attn@v into)
        add(conv("c1", "x", 64, 200)); add(conv("c2", "x", 64, 128))
        add(reshape("s", "c1", [1, 2, N, N])); add(reshape("v", "c2", [1, 2, 64, N]))
        nodes.append(h.make_node("Transpose", ["s"], ["st"], perm=[0, 1, 3, 2]))
        nodes.append(h.make_node("MatMul", ["v", "st"], ["y"]))
    elif kind == "av":  # attn @ v with a plain (no-transpose) v of shape [2,100,64]
        add(conv("c1", "x", 64, 200)); add(conv("c2", "x", 64, 128))
        add(reshape("s", "c1", [1, 2, N, N])); add(reshape("v", "c2", [1, 2, N, 64]))
        nodes.append(h.make_node("MatMul", ["s", "v"], ["y"]))
    else:
        raise ValueError(kind)
    g = h.make_graph(nodes, kind, [h.make_tensor_value_info("x", T.FLOAT, [1, 64, 10, 10])],
                     [h.make_tensor_value_info("y", T.FLOAT, None)], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 11)])
    m.ir_version = 7
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    return m


KINDS = ["t4", "t3", "nhwc", "mm", "mm3", "sm", "mmsm", "qk", "vaT", "av", "sm2", "sm4", "sm1", "mmsm3"]

if __name__ == "__main__":
    for k in KINDS:
        d = "onnx/micro/" + k
        os.makedirs(d, exist_ok=True)
        onnx.save(build(k), d + "/model.onnx")
        xs = [rng.randn(1, 64, 10, 10).astype(np.float32) * 2 for _ in range(4)]
        np.savez(d + "/calib.npz", **{"%d|x" % i: x for i, x in enumerate(xs)})
        np.save(d + "/input.npy", xs[0])
    print("built", KINDS)
