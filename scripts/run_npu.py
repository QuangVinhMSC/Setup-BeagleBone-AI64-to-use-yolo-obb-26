"""Run a compiled YOLO26-OBB TIDL model on the BeagleBone AI-64 NPU (or ARM-only with --cpu).

  cd ~/npu-test
  # eMMC Debian, TIDL 8.2
  echo temppwd | sudo -S -E env PYTHONNOUSERSITE=1 python3 run_npu.py yolo-26-obb-weight_one2one_320_8bit
  echo temppwd | sudo -S -E env PYTHONNOUSERSITE=1 python3 run_npu.py yolo-26-obb-weight_one2one_320_8bit --cpu
  # SD card, TI SDK 10 (root@192.168.1.2): no sudo/env needed
  timeout 300 python3 scripts/run_npu.py tidl_models/t10_one2one_att_8bit

Model dir = a tidl_models/<name> folder from compile_tidl.py: either a split pipeline {pipeline.json, s*.onnx,
s??_tidl/artifacts/} (recommended) or a single {model.onnx, artifacts/}. Both contain model.json.
Writes <out>/results.json (same layout as results/pytorch_baseline.json, image coords) and <out>/*.png drawings.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as rt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obb_common import draw, letterbox, postprocess, read_text, to_image_coords  # noqa: E402
from pipeline import Pipeline, cpu_session, tidl_session, tidl_session_factory  # noqa: E402


class Single(object):
    """Same interface as Pipeline for a single model.onnx + artifacts/ folder."""
    def __init__(self, d, cpu):
        path = os.path.join(d, "model.onnx")
        if cpu:
            self.sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
        else:
            self.sess = tidl_session(path, os.path.join(d, "artifacts"))
        self.input = self.sess.get_inputs()[0].name
        # TIDL 10 returns its outputs as 6-D (1,1,1,C,H,W); restore the declared ONNX shapes
        self.shapes = [o.shape for o in self.sess.get_outputs()]
        self.times = {}

    def run(self, x):
        outs = self.sess.run(None, {self.input: x})
        return [o.reshape(s) if all(isinstance(v, int) for v in s) else o for o, s in zip(outs, self.shapes)]

    def providers(self):
        return {"model": self.sess.get_providers()[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--images", default="test-images/*.png")
    ap.add_argument("--cpu", action="store_true", help="CPUExecutionProvider only (ARM reference)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=10, help="timed runs per image")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.model_dir, "model.json")))
    names = {int(k): v for k, v in meta["names"].items()}
    mode = "cpu" if a.cpu else "npu"
    out = a.out or "results/{}_{}".format(os.path.basename(os.path.normpath(a.model_dir)), mode)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    if os.path.exists(os.path.join(a.model_dir, "pipeline.json")):
        model = Pipeline(a.model_dir, cpu_session if a.cpu else tidl_session_factory(a.model_dir))
    else:
        model = Single(a.model_dir, a.cpu)
    print("session ready in %.1fs, providers %s" % (time.time() - t0, model.providers()))

    imgs = sorted(glob.glob(a.images))
    x0, _, _ = letterbox(cv2.imread(imgs[0]), meta["imgsz"])
    for _ in range(a.warmup):
        model.run(x0)

    results, t_pre, t_inf, t_post, t_stage = {}, [], [], [], {}
    for f in imgs:
        img = cv2.imread(f)
        t = time.time()
        x, r, pad = letterbox(img, meta["imgsz"])
        t_pre.append(time.time() - t)
        for _ in range(a.repeat):
            t = time.time()
            outs = model.run(x)
            t_inf.append(time.time() - t)
            for k, v in model.times.items():
                t_stage.setdefault(k, []).append(v)
        t = time.time()
        d = to_image_coords(postprocess(outs, meta["branch"], conf=a.conf, strides=meta["strides"]), r, pad)
        t_post.append(time.time() - t)

        key = os.path.basename(f)
        dets = sorted(d.tolist(), key=lambda v: (round(v[1] / 40), v[0]))
        results[key] = [(int(c), round(s, 3), [round(v, 1) for v in (x_, y_, w_, h_, th)])
                        for x_, y_, w_, h_, th, s, c in dets]
        cv2.imwrite(os.path.join(out, key), draw(img, d, names))
        print("%-45s %2d dets  %s" % (key, len(d), read_text(d, names)))

    ms = lambda v: 1000 * float(np.median(v))  # noqa: E731
    timing = {"mode": mode, "providers": model.providers(), "pre_ms": ms(t_pre), "infer_ms": ms(t_inf),
              "infer_ms_min": 1000 * float(np.min(t_inf)), "post_ms": ms(t_post),
              "stage_ms": {k: ms(v) for k, v in sorted(t_stage.items())}}
    print("median ms: pre %.2f | infer %.2f (min %.2f) | post %.2f" %
          (timing["pre_ms"], timing["infer_ms"], timing["infer_ms_min"], timing["post_ms"]))
    if timing["stage_ms"]:
        print("per stage median ms: " + ", ".join("%s %.2f" % kv for kv in timing["stage_ms"].items()))
    json.dump({"timing": timing, "dets": results}, open(os.path.join(out, "results.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
