"""Compare the two checkpoints' state dicts and head attributes."""
import torch
a, b = [torch.load(f, map_location="cpu", weights_only=False) for f in ("yolo-26-obb-weight.pt", "yolo-26-obb-weight_e2e.pt")]
ma, mb = [(c.get("ema") or c["model"]) for c in (a, b)]
sa, sb = ma.state_dict(), mb.state_dict()
print("keys equal:", sa.keys() == sb.keys(), len(sa))
diff = [k for k in sa if not torch.equal(sa[k].float(), sb[k].float())]
print("differing tensors:", len(diff), diff[:5])
ha, hb = ma.model[-1], mb.model[-1]
print({k: (getattr(ha, k, None), getattr(hb, k, None)) for k in vars(ha) if not k.startswith("_") and not isinstance(getattr(ha, k), torch.nn.Module)})
print("top-level ckpt diff:", {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if k not in ("model", "ema", "optimizer", "train_args", "train_metrics", "train_results") and a.get(k) != b.get(k)})
ta, tb = a.get("train_args", {}), b.get("train_args", {})
print("train_args diff:", {k: (ta.get(k), tb.get(k)) for k in set(ta) | set(tb) if ta.get(k) != tb.get(k)})
