"""PyTorch reference predictions at imgsz=320 (end2end on/off) for both checkpoints."""
import glob, json, os
from ultralytics import YOLO

imgs = sorted(glob.glob("test-images/*.png"))
out = {}
for w in ("yolo-26-obb-weight.pt", "yolo-26-obb-weight_e2e.pt"):
    for e2e in (False, True):
        m = YOLO(w)
        m.model.model[-1].end2end = e2e
        key = f"{os.path.splitext(w)[0]}|end2end={e2e}"
        out[key] = {}
        for f in imgs:
            r = m.predict(f, imgsz=320, conf=0.25, verbose=False)[0]
            cls = r.obb.cls.int().tolist(); xywhr = r.obb.xywhr.tolist(); conf = r.obb.conf.tolist()
            # read order: sort by row (y) then x
            dets = sorted(zip(cls, conf, xywhr), key=lambda d: (round(d[2][1] / 40), d[2][0]))
            out[key][os.path.basename(f)] = [(int(c), round(s, 3), [round(v, 1) for v in b]) for c, s, b in dets]
        txt = {k: "".join(r.names[c] if r.names[c] != "line2" else "/" for c, _, _ in v) for k, v in out[key].items()}
        print(key, "->", list(txt.values())[:4], "n_dets:", sum(len(v) for v in out[key].values()))
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/pytorch_baseline.json", "w"), indent=1)
