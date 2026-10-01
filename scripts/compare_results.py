"""Compare run_npu.py results.json files against the PyTorch baseline (image coordinates).

  python scripts/compare_results.py results/<run>/results.json [more.json ...] --ref "yolo-26-obb-weight|end2end=True"
  python scripts/compare_results.py results/<run>/results.json --baseline results/pc_float_one2one/results.json

Note: the PyTorch baseline used Ultralytics predict() (rect letterbox, not 320x320 square), so even the float ONNX
differs from it slightly. To isolate quantisation error, compare against the float ONNX run (same preprocessing).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obb_common import read_text  # noqa: E402

NAMES = {i: str(i) for i in range(10)}
NAMES.update({10: ":", 11: "M", 12: "_", 13: "line2"})


def to_arr(dets):
    """[(cls, score, [x,y,w,h,r]), ...] -> (N,7) x,y,w,h,r,score,cls."""
    return np.array([b + [s, c] for c, s, b in dets], np.float32).reshape(-1, 7)


def match(ref, d, tol):
    """Greedy nearest-center matching. Returns (#matched same class, #matched any class, max center err)."""
    if not len(ref) or not len(d):
        return 0, 0, 0.0
    dist = np.linalg.norm(ref[:, None, :2] - d[None, :, :2], axis=2)
    used, same, anyc, err = set(), 0, 0, 0.0
    for i in np.argsort(dist.min(1)):
        cand = [j for j in np.argsort(dist[i]) if j not in used]
        if cand and dist[i, cand[0]] < tol:
            j = cand[0]
            used.add(j); anyc += 1; same += int(ref[i, 6] == d[j, 6]); err = max(err, dist[i, j])
    return same, anyc, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--baseline", default="results/pytorch_baseline.json")
    ap.add_argument("--ref", default="yolo-26-obb-weight|end2end=True")
    ap.add_argument("--tol", type=float, default=15.0, help="center match tolerance in image px")
    a = ap.parse_args()
    base = json.load(open(a.baseline))
    base = base["dets"] if "dets" in base else base[a.ref]  # another run_npu.py results.json, or the PyTorch baseline

    for path in a.results:
        run = json.load(open(path))
        n_ref = n_det = n_same = n_any = n_txt = 0
        err = 0.0
        bad = []
        for img, rd in sorted(base.items()):
            ref, d = to_arr(rd), to_arr(run["dets"].get(img, []))
            same, anyc, e = match(ref, d, a.tol)
            n_ref += len(ref); n_det += len(d); n_same += same; n_any += anyc; err = max(err, e)
            t_ref, t_run = read_text(ref, NAMES), read_text(d, NAMES)
            n_txt += t_ref == t_run
            if t_ref != t_run:
                bad.append((img, t_ref, t_run))
        t = run.get("timing", {})
        print("== %s  [%s, infer %.2f ms]" % (path, t.get("mode", "?"), t.get("infer_ms", float("nan"))))
        print("   dets: ref %d, run %d | matched %d, same class %d (%.1f%%) | max center err %.1fpx"
              % (n_ref, n_det, n_any, n_same, 100.0 * n_same / max(n_ref, 1), err))
        print("   text identical: %d/%d images" % (n_txt, len(base)))
        for img, tr, tn in bad:
            print("     %s\n       ref: %s\n       run: %s" % (img, tr, tn))


if __name__ == "__main__":
    main()
