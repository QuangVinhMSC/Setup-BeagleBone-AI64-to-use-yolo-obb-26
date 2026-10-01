"""Run one small {model.onnx, artifacts/} dir on the NPU and on the CPU; print timing and max abs diff.

Used to check single ops (e.g. Sigmoid) on the target. Always wrap in `timeout`: a hung TIDL run wedges the C7x.
  timeout 60 python3 op_test.py tidl_models/t10_sig [test-images/frame.png [out.npy]]
On the PC, inside the TIDL compiler container, the same call runs TI's bit-exact emulation of the target.
"""
import os
import sys
import time

import numpy as np
import onnxruntime as rt

d = sys.argv[1]
model = os.path.join(d, "model.onnx")
art = os.path.abspath(os.path.join(d, "artifacts"))

cpu = rt.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = cpu.get_inputs()[0]
shape = [s if isinstance(s, int) else 1 for s in inp.shape]
if len(sys.argv) > 2 and sys.argv[2].endswith(".npz"):  # several named inputs
    z = np.load(sys.argv[2])
    feed = {i.name: z[i.name].astype(np.float32) for i in cpu.get_inputs()}
    x = None
elif len(sys.argv) > 2 and sys.argv[2].endswith(".npy"):  # real intermediate tensor (extract_block.py input.npy)
    x = np.load(sys.argv[2]).reshape(shape)
elif len(sys.argv) > 2:  # real image, same letterbox as calibration (random noise saturates the 8-bit ranges)
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from obb_common import letterbox
    x = letterbox(cv2.imread(sys.argv[2]), shape[2])[0].reshape(shape)
else:
    x = np.random.RandomState(0).randint(0, 256, shape)
if x is not None:
    feed = {inp.name: x.astype(np.uint8 if "uint8" in inp.type else np.float32)}
ref = cpu.run(None, feed)

t0 = time.time()
npu = rt.InferenceSession(model, providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
                          provider_options=[{"artifacts_folder": art,
                                             "debug_level": int(os.environ.get("TIDL_DEBUG", 0))}, {}])
print("%s: session %.1fs, providers %s" % (d, time.time() - t0, npu.get_providers()), flush=True)
out = npu.run(None, feed)  # first run: this is where TIDL 8.2 hung on Sigmoid
t0 = time.time()
for _ in range(20):
    out = npu.run(None, feed)
ms = (time.time() - t0) / 20 * 1e3
for r, o in zip(ref, out):
    o = o.reshape(r.shape)
    scale = float(np.abs(r).max()) or 1.0
    err = np.abs(r - o)
    print("  out %s: %.2f ms/run, max|diff| %.4f, mean|diff| %.4f, p99 %.4f (ref max %.3f)"
          % (list(r.shape), ms, float(err.max()), float(err.mean()), float(np.percentile(err, 99)), scale))
    if len(sys.argv) > 3:
        np.save(sys.argv[3], o)
print("PASS", d)
