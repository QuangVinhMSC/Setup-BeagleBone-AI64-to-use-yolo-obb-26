# setup-BeagleBone-AI64

Setup process for deploying models featuring sigmoid functions and attention blocks (YOLO26n-OBB) onto the BeagleBone AI-64 NPU (TI TDA4VM, C7x + MMA, via TIDL).

## Start here: [`docs.md`](docs.md)

**`docs.md` contains everything needed for the setup.** It covers:
- handling the board, including power and the BOOT button
- building the TI SDK 10.0 SD card
- the TIDL compilers
- converting, splitting, compiling and running the model
- a catalogue of every problem met so far, with its fix

Read it on its own, from top to bottom.

## Everything else is sample material

None of the other files are required for the setup. They are reference samples of the code and configs that `docs.md` describes:

| Path | What it is |
|---|---|
| `scripts/` | Sample Python scripts for export, splitting, compiling, running and testing. Some of them are old investigation helpers. |
| `tidl/Dockerfile`, `tidl/Dockerfile.10` | Sample Dockerfiles for the TIDL 8.2 and TIDL 10.0 compiler images. |
| `tidl/sdboot/` | Sample SD-card configs: static Ethernet, `extlinux.conf`, and the `rproc-late` service and script. |

Adapt the paths, addresses and model names in these files to your own environment.
