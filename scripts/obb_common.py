"""Pre/post-processing for the head-cut YOLO26-OBB ONNX (numpy + OpenCV only).

Runs unchanged on the PC, in the TIDL compiler container (py3.6) and on the BeagleBone AI-64 (py3.9).
"""
import math

import cv2
import numpy as np


def letterbox(img_bgr, size=320, pad=114):
    """Resize keeping aspect ratio, pad to size x size (centered). Returns NCHW float32 RGB 0..255, ratio, (dw, dh)."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # BGR->RGB, HWC->NCHW
    return np.ascontiguousarray(x), r, (left, top)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(outs, strides=(8, 16, 32)):
    """outs: list of 9 arrays [box_s8, cls_s8, ang_s8, box_s16, ...]. Returns (N,5) xywhr boxes, (N,nc) scores."""
    boxes, scores = [], []
    for i, s in enumerate(strides):
        box, cls, ang = outs[3 * i], outs[3 * i + 1], outs[3 * i + 2]
        _, _, h, w = box.shape
        box = box[0].reshape(4, -1)
        cls = cls[0].reshape(cls.shape[1], -1)
        ang = ang[0].reshape(-1)
        gy, gx = np.meshgrid(np.arange(h, dtype=np.float32) + 0.5, np.arange(w, dtype=np.float32) + 0.5, indexing="ij")
        ax, ay = gx.reshape(-1), gy.reshape(-1)
        lt, rb = box[:2], box[2:]
        xf, yf = (rb - lt) / 2
        c, sn = np.cos(ang), np.sin(ang)
        x = (xf * c - yf * sn + ax) * s
        y = (xf * sn + yf * c + ay) * s
        wh = (lt + rb) * s
        boxes.append(np.stack([x, y, wh[0], wh[1], ang], 1))
        scores.append(_sigmoid(cls).T)
    return np.concatenate(boxes), np.concatenate(scores)


def _covariance(b):
    a, bb = b[:, 2] ** 2 / 12, b[:, 3] ** 2 / 12
    c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
    return a * c ** 2 + bb * s ** 2, a * s ** 2 + bb * c ** 2, (a - bb) * c * s


def probiou(b1, b2, eps=1e-7):
    """Pairwise probabilistic IoU between (N,5) and (M,5) xywhr boxes (same as Ultralytics batch_probiou)."""
    x1, y1 = b1[:, 0:1], b1[:, 1:2]
    x2, y2 = b2[:, 0][None], b2[:, 1][None]
    a1, b1_, c1 = [v[:, None] for v in _covariance(b1)]
    a2, b2_, c2 = [v[None] for v in _covariance(b2)]
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1_ + b2_) * (x1 - x2) ** 2) / ((a1 + a2) * (b1_ + b2_) - (c1 + c2) ** 2 + eps) * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1_ + b2_) - (c1 + c2) ** 2 + eps) * 0.5
    t3 = np.log(((a1 + a2) * (b1_ + b2_) - (c1 + c2) ** 2)
                / (4 * np.sqrt(np.clip(a1 * b1_ - c1 ** 2, 0, None) * np.clip(a2 * b2_ - c2 ** 2, 0, None)) + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def _regularize(b):
    """Make w >= h and angle in [0, pi) like Ultralytics regularize_rboxes."""
    x, y, w, h, t = b.T
    swap = (t % math.pi) >= math.pi / 2
    w2, h2 = np.where(swap, h, w), np.where(swap, w, h)
    t = t % (math.pi / 2)
    return np.stack([x, y, w2, h2, t], 1)


def postprocess(outs, branch, conf=0.25, iou=0.7, max_det=300, strides=(8, 16, 32)):
    """Returns (K,7): x, y, w, h, angle(rad), score, class — in network-input (letterboxed) pixels."""
    boxes, scores = decode(outs, strides)
    cls = scores.argmax(1)
    sc = scores[np.arange(len(cls)), cls]
    if branch == "one2one":  # NMS-free: top-k by score
        order = np.argsort(-sc)[:max_det]
        keep = order[sc[order] > conf]
    else:  # one2many: class-aware rotated NMS with probIoU
        cand = np.where(sc > conf)[0]
        cand = cand[np.argsort(-sc[cand])]
        keep = []
        if len(cand):
            b = boxes[cand].copy()
            b[:, :2] += cls[cand][:, None] * 7680.0  # offset per class -> class-aware NMS
            ious = np.triu(probiou(b, b), 1)
            keep = cand[ious.max(0) < iou][:max_det]  # same "fast" rule as Ultralytics TorchNMS for OBB
    keep = np.asarray(keep, dtype=int)
    b = _regularize(boxes[keep]) if len(keep) else np.zeros((0, 5), np.float32)
    return np.concatenate([b, sc[keep, None], cls[keep, None].astype(np.float32)], 1)


def to_image_coords(dets, ratio, pad):
    d = dets.copy()
    d[:, 0] = (d[:, 0] - pad[0]) / ratio
    d[:, 1] = (d[:, 1] - pad[1]) / ratio
    d[:, 2:4] /= ratio
    return d


def read_text(dets, names):
    """Group detections into text lines by y, left-to-right. 'line2' boxes are line markers and are skipped."""
    chars = [d for d in dets if names[int(d[6])] != "line2"]
    if not chars:
        return ""
    chars.sort(key=lambda d: d[1])
    med_h = float(np.median([min(d[2], d[3]) for d in chars]))
    lines, cur = [], [chars[0]]
    for d in chars[1:]:
        if abs(d[1] - np.mean([c[1] for c in cur])) > 0.6 * med_h:
            lines.append(cur)
            cur = [d]
        else:
            cur.append(d)
    lines.append(cur)
    return " / ".join("".join(names[int(c[6])] for c in sorted(l, key=lambda c: c[0])) for l in lines)


def draw(img, dets, names):
    for x, y, w, h, t, s, c in dets:
        pts = cv2.boxPoints(((float(x), float(y)), (float(w), float(h)), math.degrees(float(t)))).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.putText(img, "%s %.2f" % (names[int(c)], s), (int(pts[:, 0].min()), int(pts[:, 1].min()) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return img
