"""Run a split model (pipeline.json from split_tidl.py): TIDL parts and CPU parts executed in order.

Runs unchanged on the PC, in the TIDL compiler container (py3.6) and on the board (py3.9). numpy + onnxruntime only.
"""
import json
import os
import time

import numpy as np
import onnxruntime as rt


# TIDL 8.2 ships onnxruntime 1.7 (eMMC Debian); TIDL 10.0 ships 1.14 (SD card, TI SDK 10)
TIDL10 = not rt.__version__.startswith("1.7")


def cpu_session(stage, path):
    return rt.InferenceSession(path, providers=["CPUExecutionProvider"])


def tidl_session(path, art, debug_level=0):
    """TIDL EP session for a compile_tidl.py artifacts folder, with the options of the installed TIDL version."""
    opts = {"artifacts_folder": os.path.abspath(art), "debug_level": debug_level}
    so = rt.SessionOptions()
    if TIDL10:  # graph must match the one compiled (compile_tidl.py uses ORT_DISABLE_ALL too)
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        opts.update({"platform": "J7", "version": "8.2"})
    return rt.InferenceSession(path, so, providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
                               provider_options=[opts, {}])


def tidl_session_factory(root, debug_level=0):
    """TIDL parts use <root>/<stage name>/artifacts (compile_tidl.py layout); CPU parts run on the ARM."""
    def make(stage, path):
        if stage["kind"] != "tidl":
            return cpu_session(stage, path)
        return tidl_session(path, os.path.join(root, stage["name"], "artifacts"), debug_level)
    return make


class Pipeline(object):
    def __init__(self, d, make_session=cpu_session):
        self.dir = d
        self.spec = json.load(open(os.path.join(d, "pipeline.json")))
        self.stages = [(s, make_session(s, os.path.join(d, s["model"]))) for s in self.spec["stages"]]
        # TIDL 10 returns outputs as 6-D (1,1,1,C,H,W); remember the declared ONNX shapes to restore them
        self.shapes = [{o.name: o.shape for o in sess.get_outputs()} for _, sess in self.stages]
        self.times = {}

    def run_all(self, x):
        """Returns every tensor that crosses a stage boundary (dict name -> array)."""
        env = {self.spec["input"]: x.astype(self.spec.get("input_dtype", "float32"))}
        for (s, sess), shapes in zip(self.stages, self.shapes):
            t = time.time()
            outs = sess.run([p for p, _ in s["outputs"]], {i: env[i] for i in s["inputs"]})
            self.times[s["name"]] = time.time() - t
            for (p, name), v in zip(s["outputs"], outs):
                shp = shapes[p]
                env[name] = v.reshape(shp) if all(isinstance(d, int) for d in shp) else v
        return env

    def run(self, x):
        env = self.run_all(x)
        return [env[o] for o in self.spec["outputs"]]

    def providers(self):
        return {s["name"]: sess.get_providers()[0] for s, sess in self.stages}
