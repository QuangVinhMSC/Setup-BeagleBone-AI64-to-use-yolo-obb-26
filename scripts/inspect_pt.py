"""Print metadata of the Ultralytics checkpoints."""
import sys, torch
for f in sys.argv[1:]:
    ck = torch.load(f, map_location="cpu", weights_only=False)
    m = ck.get("ema") or ck["model"]
    h = m.model[-1]
    print("=====", f)
    print(" nc:", m.nc, "names:", m.names)
    print(" yaml:", {k: v for k, v in m.yaml.items() if k not in ("backbone", "head")})
    print(" imgsz:", ck.get("train_args", {}).get("imgsz"), "ultralytics:", ck.get("version"))
    print(" head:", type(h).__name__, {k: getattr(h, k) for k in ("end2end", "max_det", "reg_max", "ne") if hasattr(h, k)})
