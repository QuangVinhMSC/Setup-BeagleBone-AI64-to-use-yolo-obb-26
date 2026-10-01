"""Compile a head-cut ONNX, or a split pipeline, into TIDL 8.2 artifacts (run INSIDE the tidl-8.2-compiler container).

  # split pipeline (recommended; see split_tidl.py for why)
  docker run --rm -v "D:/somethings/NPU-TEST:/work" tidl-8.2-compiler \
      python3 scripts/compile_tidl.py onnx/yolo-26-obb-weight_one2one_320_split --bits 8
  -> tidl_models/<name>_<bits>bit/{pipeline.json, model.json, s*.onnx, s??_tidl/artifacts/}

  # single ONNX with TIDL's own partitioning (deny list). Hangs on the target for this model, kept for reference.
  docker run ... python3 scripts/compile_tidl.py onnx/yolo-26-obb-weight_one2one_320.onnx --bits 8
  -> tidl_models/<name>_<bits>bit/{model.onnx, model.json, artifacts/}

The output folder is what gets copied to the board.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import onnxruntime as rt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obb_common import letterbox  # noqa: E402


# TIDL 8.2 ships onnxruntime 1.7 (tidl-8.2-compiler); TIDL 10.0 ships 1.14 (tidl-10.0-compiler, Dockerfile.10)
TIDL10 = not rt.__version__.startswith("1.7")


def tidl_options(a, art, n_frames, deny):
    o = {
        "tidl_tools_path": os.environ["TIDL_TOOLS_PATH"],
        "artifacts_folder": art,
        "tensor_bits": a.bits,
        "debug_level": int(os.environ.get("TIDL_DEBUG", 0)),
        "max_num_subgraphs": a.max_subgraphs,
        "accuracy_level": 1,
        "advanced_options:calibration_frames": n_frames,
        "advanced_options:calibration_iterations": a.calib_iters,
        "advanced_options:output_feature_16bit_names_list": "",
        "advanced_options:params_16bit_names_list": "",
        "advanced_options:quantization_scale_type": a.qscale,
        "advanced_options:high_resolution_optimization": 0,
        "advanced_options:pre_batchnorm_fold": 1,
        "advanced_options:activation_clipping": 1,
        "advanced_options:weight_clipping": 1,
        "advanced_options:bias_calibration": 1,
        "advanced_options:add_data_convert_ops": a.data_convert,
        "advanced_options:channel_wise_quantization": 0,
    }
    if TIDL10:
        o["deny_list:layer_type"] = deny
    else:
        o.update({"platform": "J7", "version": "8.2", "ti_internal_nc_flag": 1601, "deny_list": deny})
    return o


def compile_model(a, model, art, feeds, deny):
    """feeds: list of {input name: array}, one per calibration frame."""
    os.makedirs(art)
    so = rt.SessionOptions()
    if TIDL10:  # TI: transformer (MatMul/Softmax) import needs ORT graph optimisations off
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = rt.InferenceSession(model, so, providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
                               provider_options=[tidl_options(a, art, len(feeds), deny), {}])
    dtype = {i.name: np.uint8 if "uint8" in i.type else np.float32 for i in sess.get_inputs()}
    for f in feeds:  # calibration pass: each run() feeds one frame to the TIDL quantiser
        sess.run(None, {k: v.astype(dtype[k]) for k, v in f.items()})
    del sess  # artifacts are written when the session is released
    print("artifacts in", art, ":", sorted(x for x in os.listdir(art) if x != "tempDir"))


def calib_images(a, meta):
    return [letterbox(cv2.imread(f), meta["imgsz"])[0] for f in sorted(glob.glob(a.calib))]


def compile_pipeline(a):
    from pipeline import Pipeline
    src = a.onnx.rstrip("/")
    meta = json.load(open(os.path.join(src, "model.json")))
    out = a.out or "tidl_models/{}_{}bit".format(os.path.basename(src), a.bits)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, out)

    # calibration inputs of every TIDL part = float pipeline tensors on the calibration images
    pipe = Pipeline(src)
    envs = [pipe.run_all(x) for x in calib_images(a, meta)]
    for s in pipe.spec["stages"]:
        if s["kind"] != "tidl":
            continue
        feeds = os.path.join(out, s["name"] + "_calib.npz")
        np.savez(feeds, **{"%d|%s" % (k, i): e[i] for k, e in enumerate(envs) for i in s["inputs"]})
        # one process per part: the TIDL compiler segfaults on teardown after writing the artifacts
        cmd = [sys.executable, os.path.abspath(__file__), os.path.join(out, s["model"]), "--feeds", feeds,
               "--out", os.path.join(out, s["name"]), "--bits", str(a.bits), "--calib-iters", str(a.calib_iters),
               "--data-convert", str(a.data_convert), "--qscale", str(a.qscale), "--deny", ""]
        print("== compiling", s["name"], flush=True)
        subprocess.call(cmd)
        os.remove(feeds)
        if not glob.glob(os.path.join(out, s["name"], "artifacts", "*_tidl_net.bin")):
            sys.exit("compile of %s produced no artifacts" % s["name"])
    print("pipeline compiled ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="head-cut .onnx, split pipeline dir, or (with --feeds) one part .onnx")
    ap.add_argument("--bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--calib", default="test-images/*.png", help="glob of calibration images")
    ap.add_argument("--calib-iters", type=int, default=5)
    # Single-ONNX mode only: these op types only occur inside the two attention blocks (plus one Split right before
    # one). TIDL 8.2 cannot run MatMul, and if it grabs the Split/Reshape pieces around it the import crashes or
    # misreads the Reshape as a Flatten. So the whole attention core runs on the ARM.
    ap.add_argument("--deny", default="MatMul,Softmax,Transpose,Reshape,Split",
                    help="comma separated ONNX op types to keep on ARM")
    ap.add_argument("--max-subgraphs", type=int, default=16)
    # 3 = in/out float<->int conversion runs on the C7x as extra layers; 0 = done by the runtime on the ARM
    ap.add_argument("--data-convert", type=int, default=3, choices=[0, 1, 2, 3])
    # 0 = non-power-of-2 scales (TI example default), 1 = power-of-2 (what TI used for the model-zoo artifacts)
    ap.add_argument("--qscale", type=int, default=0, choices=[0, 1])
    ap.add_argument("--feeds", default=None, help="(internal) .npz of calibration inputs for one pipeline part")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if os.path.isdir(a.onnx):
        return compile_pipeline(a)
    if a.feeds:  # one part of a pipeline, called by compile_pipeline()
        z = np.load(a.feeds)
        n = 1 + max(int(k.split("|")[0]) for k in z.files)
        feeds = [{k.split("|", 1)[1]: z[k] for k in z.files if k.startswith("%d|" % i)} for i in range(n)]
        return compile_model(a, a.onnx, os.path.join(a.out, "artifacts"), feeds, a.deny)

    meta = json.load(open(a.onnx[:-5] + ".json"))
    name = os.path.basename(a.onnx)[:-5]
    out = a.out or "tidl_models/{}_{}bit".format(name, a.bits)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    shutil.copy(a.onnx, os.path.join(out, "model.onnx"))
    shutil.copy(a.onnx[:-5] + ".json", os.path.join(out, "model.json"))
    feeds = [{meta["input"]: x} for x in calib_images(a, meta)]
    compile_model(a, os.path.join(out, "model.onnx"), os.path.join(out, "artifacts"), feeds, a.deny)


if __name__ == "__main__":
    main()
