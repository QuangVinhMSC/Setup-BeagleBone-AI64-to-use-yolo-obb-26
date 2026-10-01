"""Check head-cut ONNX (float, onnxruntime CPU) + numpy postprocess against PyTorch on identical 320x320 input."""
import glob, json, os, sys
import cv2, numpy as np, onnxruntime as ort, torch
sys.path.insert(0, os.path.dirname(__file__))
from obb_common import letterbox, postprocess

CASES = (("onnx/yolo-26-obb-weight_one2many_320.onnx", "yolo-26-obb-weight.pt"),
         ("onnx/yolo-26-obb-weight_one2one_320.onnx", "yolo-26-obb-weight.pt"),
         ("onnx/yolo-26-obb-weight_e2e_one2one_320.onnx", "yolo-26-obb-weight_e2e.pt"))


def torch_ref(model, x, branch):
    from ultralytics.utils.nms import non_max_suppression
    model.model[-1].end2end = branch == "one2one"
    with torch.no_grad():
        y = model(torch.from_numpy(x) / 255)
    y = y[0] if isinstance(y, (tuple, list)) else y
    if branch == "one2one":  # (1,300,7): x,y,w,h,score,cls,angle
        d = y[0][y[0][:, 4] > 0.25].numpy()
        return np.concatenate([d[:, :4], d[:, 6:7], d[:, 4:6]], 1)
    d = non_max_suppression(y, 0.25, 0.7, nc=14, rotated=True)[0].numpy()  # x,y,w,h,score,cls,angle
    return np.concatenate([d[:, :4], d[:, 6:7], d[:, 4:6]], 1)


def match(a, b):
    """Greedy nearest-center matching; returns #matched with same class and max center error."""
    if not len(a) or not len(b):
        return 0, 0.0
    dist = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
    used, ok, err = set(), 0, 0.0
    for i in np.argsort(dist.min(1)):
        j = int(np.argmin(np.where(np.isin(np.arange(len(b)), list(used)), 1e9, dist[i])))
        if dist[i, j] < 8 and j not in used:
            used.add(j); ok += int(a[i, 6] == b[j, 6]); err = max(err, dist[i, j])
    return ok, err


if __name__ == "__main__":
    for onnx_path, pt in CASES:
        meta = json.load(open(onnx_path[:-5] + ".json"))
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ck = torch.load(pt, map_location="cpu", weights_only=False); model = (ck.get("ema") or ck["model"]).float().eval()
        n_ref = n_ok = 0; err = 0.0
        for f in sorted(glob.glob("test-images/*.png")):
            x, _, _ = letterbox(cv2.imread(f), 320)
            d = postprocess(sess.run(None, {"images": x}), meta["branch"])
            ref = torch_ref(model, x, meta["branch"])
            ok, e = match(ref, d); n_ref += len(ref); n_ok += ok; err = max(err, e)
        print(f"{os.path.basename(onnx_path)}: {n_ok}/{n_ref} PyTorch dets matched (same class), max center err {err:.3f}px")
