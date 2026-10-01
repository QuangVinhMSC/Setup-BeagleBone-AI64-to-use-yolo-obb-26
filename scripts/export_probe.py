"""Probe Ultralytics ONNX export at 320 with/without end2end; list op types and outputs."""
import collections, shutil, onnx
from ultralytics import YOLO
for e2e in (False, True):
    m = YOLO("yolo-26-obb-weight.pt")
    p = m.export(format="onnx", imgsz=320, opset=11, simplify=True, dynamic=False, end2end=e2e)
    g = onnx.load(p).graph
    print(f"--- end2end={e2e}: outputs", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output])
    print(dict(collections.Counter(n.op_type for n in g.node)))
    shutil.move(p, f"scratch_probe_e2e{e2e}.onnx")
