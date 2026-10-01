"""Run the raw PyTorch head in both branches for both checkpoints; show class histograms."""
import sys, os, collections, torch, cv2, numpy as np
sys.path.insert(0, "scripts"); from obb_common import letterbox
x, _, _ = letterbox(cv2.imread("test-images/frame_000001_20260909_155356_135520.png"))
x = torch.from_numpy(x) / 255
for w in ("yolo-26-obb-weight.pt", "yolo-26-obb-weight_e2e.pt"):
    ck = torch.load(w, map_location="cpu", weights_only=False); m = (ck.get("ema") or ck["model"]).float().eval()
    for e2e in (False, True):
        m.model[-1].end2end = e2e
        with torch.no_grad(): y = m(x)
        y = y[0] if isinstance(y, (tuple, list)) else y
        if e2e:  # (1,300,7) x1..,score,cls,angle
            d = y[0][y[0][:, 4] > 0.25]; cls = d[:, 5].int().tolist()
        else:  # (1, 4+nc+1, N)
            s = y[0, 4:18].T; conf, c = s.max(1); cls = c[conf > 0.25].tolist()
        print(w, "end2end", e2e, "n>0.25:", len(cls), dict(sorted(collections.Counter(cls).items())))
