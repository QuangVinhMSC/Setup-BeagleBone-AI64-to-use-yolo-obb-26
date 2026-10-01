# Running YOLO26n-OBB on the BeagleBone AI-64 NPU: setup, problems and solutions

These notes describe how to get a modern YOLO model (YOLO26n-OBB) running on the BeagleBone AI-64's NPU. The board is built on a TI TDA4VM: 2× Cortex-A72, a C7x DSP with MMA (the "NPU"), 2× C66x and 6× R5F. The notes cover the hardware and boot quirks, the two TI software stacks (SDK 8.2 and SDK 10.0), the model conversion pipeline, and the problems met along the way with their fixes.

The document is meant to be self-contained. Where to look:

| I want to… | Read |
|---|---|
| Understand how the whole thing works | §11 |
| Run a finished model on the board | §12 |
| Set up a new PC / rebuild the compilers | §13 |
| Convert a new or retrained checkpoint | §14 |
| Use the model from my own Python code | §15 |
| Fix bad accuracy, test a new op, debug a failure | §16, then §7 (problem catalogue) |
| Handle the board (power, BOOT button, SD card) | §2, §3 |
| Build the SDK 10 SD card | §6 |

Placeholders used throughout: `<project>` is the project root on the PC, `<python>` the Python interpreter of the conversion environment, `<workdir>` a working directory on the board, `<model_dir>` a compiled model folder, `<onnx>` an exported ONNX file, `<split_dir>` a split pipeline folder.

---

## 1. Overview

| Item | Description |
|---|---|
| PC | Windows, with a POSIX shell (e.g. Git Bash) and PowerShell, a conda Python environment, Docker Desktop |
| Board link 1 | **USB-C**: power *and* USB networking (RNDIS). Board `192.168.7.2` / `192.168.6.2`, PC `192.168.7.1` / `192.168.6.1` (BeagleBoard defaults) |
| Board link 2 | **Ethernet cable straight to the PC** (no router, no DHCP). Static addresses on both ends, e.g. PC `192.168.1.100/24`, board `192.168.1.2` |
| Extra power | A **barrel-jack adapter** may also be connected (see §2.1, this matters) |
| eMMC (16 GB) | Stock BeagleBoard Debian 11 Bullseye image, kernel 5.10-ti, **TI SDK 8.2** |
| microSD | **TI Processor SDK Linux SK-TDA4VM 10.00.00.08** (Edge AI), adapted for the BBAI64 (§6) |

Two operating systems can live on the board. Which one runs depends on the SD card and the BOOT button (§3).

| System | Reach it at | Login | TIDL | NPU usable for YOLO26? |
|---|---|---|---|---|
| eMMC Debian 11 | `ssh debian@192.168.7.2` (USB) | key auth; default `debian` password | 8.2 | ❌ Sigmoid/SiLU hangs the C7x |
| SD TI SDK 10.0 | `ssh root@<board-eth-ip>` (Ethernet) | key auth; root has no password | 10.0 | ✅ whole model except Softmax, ~19 ms/inference (§9) |

---

## 2. Physical handling

### 2.1 Power: two sources, and why it matters
- The board runs whenever **any** power source is connected. **USB-C** and the **barrel adapter** both power it.
- **To really power it off, unplug both.** Then wait until every LED is dark (~5 s).
  - Unplugging only USB-C while the adapter stays in does **nothing**: the board keeps running.
  - This is an easy trap: a "hold BOOT and plug in USB-C" attempt then never actually restarts the board, the new bootloader is never tested, and debugging goes after the wrong theory.
- **Power on:** plug the power back in. It starts by itself; LEDs light within a few seconds. If they don't, briefly press **POWER**.
- **Clean shutdown first when possible** (`sudo poweroff` on eMMC Debian; `sync; poweroff` as root on the SD system), then unplug.
- Pulling power on a running board is electrically safe. The risk is filesystem damage on whatever is mounted read-write:
  - ext4's journal normally recovers on the next mount.
  - The SD system **auto-mounts the eMMC rootfs read-write** (§7). Unmount it before pulling power.
- Boot time: eMMC Debian answers after about 1–2 min. The SD system answers SSH after about 25 s (plus 1–2 min on its very first boot).

### 2.2 BOOT button: selecting the SD card's bootloader
- **Holding BOOT while power is applied from a fully off board** makes the ROM load the bootloader from the **microSD** (`tiboot3.bin` on its FAT partition).
- It only works from a fully off state (§2.1). Procedure:
  1. Unplug the adapter and USB-C. Wait for the LEDs to go dark.
  2. Press and **hold BOOT**.
  3. Still holding, plug in the adapter, then USB-C. Keep holding ~5 s, then release.
     - No LEDs? Keep holding BOOT, tap **POWER**, release BOOT after ~5 s.
  4. Wait ~1–2 min.
- A **warm `reboot`** from the SD system keeps the SD bootloader's firmware; BOOT is only needed after a cold power-on. This is the quick way to recover a wedged C7x.
- Verify which firmware came up (on the SD system): `dmesg | grep "firmware rev"`
  - `11.1.8--v11.01.08` means the SD bootloader ran ✅ (NPU works).
  - `21.9.1--v2021.09a` means the eMMC bootloader ran ❌ (NPU firmware stalls, §7).

### 2.3 microSD card
- The card holds the TI SDK 10.0 system. **Without** BOOT held and **with** the card inserted, the eMMC's own U-Boot still boots the SD kernel: it prefers an `extlinux.conf` on the SD. But that path uses the old firmware and the NPU won't work.
- **To get the eMMC Debian back:** power off, **remove the card**, power on.
- **Hot-insert is fine.** With eMMC Debian running, push the card in and it appears as `/dev/mmcblk1` within seconds. This is the recovery route for editing a card that no longer boots (§7).
- Don't remove the card while the SD system is running.

### 2.4 Ethernet
- With a direct cable there is no router and no DHCP. Give the PC NIC a static address, and give the SD system a static address through a `systemd-networkd` file (e.g. `/etc/systemd/network/05-eth0-static.network`).
- The stock eMMC Debian doesn't configure `eth0` with an IPv4 address in this setup (link up, no address). Use USB for it.

### 2.5 Serial console (recommended)
- Several boot problems can only be debugged blind without it. A **3.3 V USB-UART adapter** (FTDI/CP2102, 115200 8N1) on the board's UART debug header shows U-Boot and kernel output. The SD system's console is `ttyS2`.
- Use only a **3.3 V** adapter.

---

## 3. Boot modes (what actually boots)

| SD card | BOOT held at power-on | Bootloader / SYSFW | Kernel + rootfs | NPU |
|---|---|---|---|---|
| out | n/a | eMMC U-Boot 2021.10 (Beagle), SYSFW 21.9.1 | eMMC Debian 11, SDK 8.2 | 8.2: no Sigmoid |
| in | no | eMMC U-Boot 2021.10, **SYSFW 21.9.1** | **SD** SDK 10.0 (via SD `extlinux.conf`) | ❌ vision-apps stalls |
| in | **yes** | **SD** U-Boot 2025.07 (Beagle), **SYSFW 11.1.8** | SD SDK 10.0 | ✅ |

---

## 4. Connecting from a Windows PC

- SSH keys: Windows has no `ssh-copy-id`; a few lines of **paramiko** can install the public key on the eMMC Debian. For the SD system, write the key straight into the SD rootfs (`/root/.ssh/authorized_keys`).
- A new IP is a new host key. `BatchMode=yes` then fails with `Host key verification failed`. Connect once with `-o StrictHostKeyChecking=accept-new`.
- **The USB RNDIS link is flaky under heavy I/O.** During large transfers (e.g. writing an SD image through the board):
  - SSH sessions get `Connection reset by peer`.
  - The PC's USB network adapter drops to a `169.254.x.x` address.
  - Board `dmesg` shows `rndis_msg_parser: unknown RNDIS message ... RNDIS command error`.

  Rules that follow:
  - Run anything long on the board **detached**, writing to a file, and poll with short SSH calls. For example: `nohup sh -c "... > /tmp/out" &`.
  - Try the other USB address (`.6.2` vs `.7.2`) if one stops answering.
  - Prefer Ethernet once the SD system runs.
- `echo <pw> | sudo -S cmd` makes the **pipe** `cmd`'s stdin. So `tar c ... | ssh board 'echo <pw> | sudo -S tar x'` silently receives nothing ("This does not look like a tar archive"). Unpack as the login user, or authenticate sudo in an earlier command.
- PC tooling gaps on Windows:
  - No `ar`, `7z` or `strings` in Git Bash. `.deb` files can be unpacked with a few lines of Python (ar format + `tarfile`).
  - `pktmon` needs admin. To see whether the board sends anything at all, compare `Get-NetAdapterStatistics -Name Ethernet` before and after 30 s (no admin needed).
  - `conda run` breaks multi-line `-c` arguments. Call the environment's `python.exe` directly.
  - Docker from Git Bash needs `MSYS_NO_PATHCONV=1`, or it receives mangled Windows paths.
  - Docker Desktop may be installed per-user (under the user's `AppData\Local\Programs`) rather than under Program Files.

---

## 5. State of the stock eMMC system

| Finding | Consequence / handling |
|---|---|
| Debian 11 image with TI **SDK 8.2** (`ti-tidl-8.2`, `onnxruntime-tidl 1.7.0`, `ti-vision-apps-8.2`) | Artifacts must come from **TIDL 8.2** tools (a dedicated compiler container). Other versions won't load. |
| eMMC root nearly full (~1 GB free) | Nothing big fits. Use the SD card for the SDK 10 system. |
| A user-installed **numpy 2.x** (+ OpenCV) in `~/.local` | Breaks onnxruntime 1.7 (`_ARRAY_API not found`). Run with `PYTHONNOUSERSITE=1`; no need to delete it. |
| `/dev/rpmsg*` is root-only | TIDL without root segfaults (exit 139). Use `sudo`. |
| No passwordless sudo | `echo <pw> \| sudo -S ...` |
| `apt` (bullseye) offers TI packages only up to **8.6** | No official route to TIDL ≥ 9 on this image. |
| NPU sanity check | A TI model-zoo model (e.g. YOLOX-S) runs in ~11 ms. |

Standard way to run TIDL on the eMMC system:
```bash
echo <pw> | sudo -S -E env PYTHONNOUSERSITE=1 timeout 60 python3 script.py
```

---

## 6. Building the TI SDK 10.0 SD card

**Why:** TIDL 8.2's Sigmoid hangs the C7x (§7), and YOLO26 has 33 SiLU = x·sigmoid(x). SDK ≥ 9.1 also adds transformer ops (MatMul/Softmax).

**Why this route:**
- No BeagleBoard image ships TIDL ≥ 9. The bookworm/trixie repos have no TIDL/edgeai/vision-apps packages at all, and the newest BBAI64 images (Debian 13.x) have none either.
- The working community route (discussed on the BeagleBoard forum) is TI's SK-TDA4VM image plus BBAI64 boot pieces.

Steps:

1. **Download** TI's SK-TDA4VM Edge AI image, `tisdk-edgeai-image-j721e-evm.wic.xz`, version 10.00.00.08 (~1 GB, direct link from TI, no login). Check it with `xz -t`.

2. **Write it from the board itself** (no SD writer needed; the eMMC is untouched). About 15 min at ~12 MB/s:
   ```bash
   cat tisdk-edgeai-image-j721e-evm.wic.xz | ssh debian@192.168.7.2 \
     'test "$(findmnt -n -o SOURCE /)" = /dev/mmcblk0p2 || exit 9; xz -dc | sudo dd of=/dev/mmcblk1 bs=4M conv=fsync'
   ```
   - The guard refuses to run unless the board booted from eMMC.
   - Verify by comparing `xz -dc image | sha256sum` on the PC with `head -c <image size> /dev/mmcblk1 | sha256sum` on the board. Run the board side detached (RNDIS flakiness, §4).

3. **BBAI64 bootloader.** Use BeagleBoard's package `bb-u-boot-beagleboneai64` 2025.07 (from the rcn-ee trixie repository):
   - Unpack the `.deb` (on Windows: with Python, §4); the files are under `opt/u-boot/bb-u-boot-beagleboneai64/`.
   - On the SD FAT partition: move TI's `tiboot3.bin tispl.bin u-boot.img sysfw.itb uEnv.txt` aside (keep them in a backup folder), then copy in Beagle's `tiboot3.bin tispl.bin u-boot.img sysfw.itb`.
   - TI's own boot files are for the SK board (its DDR config) and won't work on the BBAI64.

4. **Kernel and device tree.** TI's SDK 10 rootfs already contains `boot/dtb/ti/k3-j721e-beagleboneai64.dtb`.
   - Its reserved-memory map matches the SK board's, except for the SK's 512 MB CMA pool.
   - TI's `k3-j721e-edgeai-apps.dtbo` applies cleanly to it:
     ```bash
     fdtoverlay -i k3-j721e-beagleboneai64.dtb -o k3-j721e-beagleboneai64-edgeai.dtb k3-j721e-edgeai-apps.dtbo
     ```
   - Copy `Image` (dereference the symlink), both DTBs, and an `extlinux/extlinux.conf` to the FAT partition. The default entry:
     ```
     kernel /Image
     fdt /k3-j721e-beagleboneai64-edgeai.dtb
     append console=ttyS2,115200n8 earlycon=ns16550a,mmio32,0x02800000 root=/dev/mmcblk1p2 rw rootfstype=ext4 rootwait cma=512M
     ```
     `cma=512M` fails to reserve (§7); harmless so far.

5. **Rootfs additions** (mount `mmcblk1p2` from eMMC Debian):
   - A static Ethernet config: `/etc/systemd/network/05-eth0-static.network` with `Address=<board-ip>/24`, `DHCP=no`.
   - `/root/.ssh/authorized_keys` (the TI image runs `dropbear`; root has no password).
   - **rproc-late**, required (§7, remote-core bring-up):
     - `/etc/modprobe.d/rproc-late.conf` blacklists `ti_k3_r5_remoteproc` and `ti_k3_dsp_remoteproc`.
     - `/etc/rproc-late.sh` and `/etc/systemd/system/rproc-late.service` (**Type=simple**, enabled) modprobe R5 first, then DSP, with a `timeout` per step. They log to a file under `/root`.
     - TI's image has no `/usr/local/bin`, so the script lives in `/etc`.

6. **Boot with BOOT held** (§2.2). Healthy state:
   ```bash
   dmesg | grep "firmware rev"                      # 11.1.8
   for r in /sys/class/remoteproc/*; do echo $(cat $r/name) $(cat $r/state); done
   #  5c00000.r5f, 5d00000.r5f, both 4d8x800000.dsp (C66) and 64800000.dsp (C7x): running
   ls /sys/bus/rpmsg/devices | grep -c chrdev        # 11
   ```

7. **Compiler for it:** a Docker image with:
   - Ubuntu 22.04, `edgeai-tidl-tools` **10_00_02_00**, SOC `am68pa` (= TDA4VM/J721E).
   - onnxruntime-tidl `1.14.0+10000005`, the same build as on the SD rootfs.

The SD system as found after boot: TI "Arago 2023.10", kernel 6.6.32-ti, python 3.12, numpy 1.26.4, OpenCV 4.9. No `sudo` or `PYTHONNOUSERSITE` needed (you are root). Hostname `j721e-sk`. The clock starts in 2024, so `tar` warns about timestamps "in the future" (harmless).

---

## 7. Problem catalogue

Legend: ✅ solved · ⚠️ workaround · ❌ unsolved / by design.

### Connection and base system

| Problem | Symptom | Cause | Fix |
|---|---|---|---|
| `conda run` breaks | Multi-line `-c` scripts fail | conda quoting | ✅ Call the env's `python.exe` directly |
| TIDL needs root | Segfault, exit 139 | `/dev/rpmsg*` root-only | ✅ Run with `sudo` |
| numpy clash | `_ARRAY_API not found` | numpy 2.x in `~/.local` | ✅ `PYTHONNOUSERSITE=1` |
| NPU wedged after a hang | Even a TI zoo model hangs afterwards | C7x firmware stuck | ✅ Reboot, wait for a small `/proc/uptime`. **Always wrap NPU runs in `timeout`** |
| `MEM: Alloc failed status=12`, `Verify OpenVX graph failed` | ENOMEM on the 2nd+ TIDL graph | Not real OOM. A wedged C7x, or `debug_level 3` trace buffers | ✅ Reboot. Keep `debug_level` ≤ 1 except for C7x layer traces |

### Model conversion and TIDL 8.2 (eMMC)

| Problem | Symptom | Cause | Fix |
|---|---|---|---|
| Importer crash | `*** buffer overflow detected ***` | Long tensor names; MaxPool 5×5/s1 unsupported | ✅ Shorten tensor names; SPPF 5×5 → 2× 3×3 |
| Attention can't go to TIDL 8.2 | Import segfault; Reshape misread as Flatten | No MatMul in 8.2 | ⚠️ Deny `MatMul,Softmax,Transpose,Reshape,Split` → attention on the ARM |
| A checkpoint's one2one head broken | Every detection class 0, also in PyTorch | The checkpoint itself | ❌ Use a checkpoint whose one2one head works; the broken one needs retraining |
| MSMC warning at compile | `TIDL_E_DATAFLOW_INFO_NULL ...` | Benign (first "float import" pass only) | ✅ Ignore |
| Compiler exits 139 at the end | `free(): invalid pointer` | Teardown crash after artifacts are written | ✅ Compile each part in its own subprocess |
| PyTorch baseline ≠ float ONNX | Lower class match, many differing texts | The PyTorch baseline used a non-square letterbox | ✅ Compare against a float ONNX run with the same preprocessing |
| `add_data_convert_ops` 0/1 | Flattened `{1,1,1,25600}` tensors | 8.2 hands back flattened outputs | ✅ Keep 3 |
| `max_num_subgraphs` ignored | Still 4 subgraphs | 8.2 limitation | ⚠️ Explicit split pipeline |
| **First `sess.run()` never returns** | Timeout, C7x wedged | **TIDL 8.2 `Sigmoid` hangs on this board** in any mode (8/16-bit, pow2 scales). HardSigmoid/Tanh not offloaded. SiLU→HardSwish without retraining: no text read correctly | ✅ Move to SDK 10 (below). Alternatives: retrain with ReLU, or ARM-only (208 ms/frame) |

### Getting to SDK 10 (BBAI64 + TI's SK image)

| Problem | Symptom | Cause | Fix |
|---|---|---|---|
| No TIDL ≥ 9 for BBAI64 | No packages, no images | BeagleBoard stopped at SDK 8.x packaging | ✅ TI SK-TDA4VM image + Beagle bootloader + BBAI64 DTB (§6) |
| Unplugging USB-C doesn't power off | BOOT-held "restart" does nothing | Barrel adapter still powering the board | ✅ Unplug **both** (§2.1) |
| First SD boot looks "dead" | Ethernet link up, no ping/ARP/IPv6, 0 packets | Board never actually restarted (previous row), or a slow first boot (setup) | ✅ Correct power cycle; allow 2+ min on first boot |
| **All TIDL runs fail on SDK 10**, even TI's zoo models | `TIDL Compute Invoke Failed`; ARM log `IPC: ERROR: Unable to create TX channels for CPU [mcu2_0] [mcu2_1] [c6x_1] [c6x_2] [c7x_1]`; empty remote log and `debugfs` traces | The eMMC U-Boot booted the SD kernel with **SYSFW 21.9.1 (2021)**; SDK 10 firmware stalls silently. See also the next row | ✅ Boot the SD's own bootloader: **hold BOOT** (SYSFW 11.1.8) |
| Main R5F core1 never starts | `Timed out waiting for 5c00000.r5f core to power up!`; no `5d00000` remoteproc | TI U-Boot pre-boots the main R5Fs (`dorprocboot=1`), Beagle's doesn't. The 6.6 split-mode driver waits only 2 s for core0 | ✅ `rproc-late`: blacklist rproc modules, load R5 then DSP after the rootfs is up |
| Long SSH sessions die | `Connection reset by peer`, adapter falls to `169.254.x.x` | Windows RNDIS under heavy I/O | ✅ Detached jobs + polling; alternate USB IP; Ethernet |
| **SD boot stalls forever** after adding a service | Ping OK, SSH `Connection refused` | Service was a blocking `oneshot` (no timeout) and a modprobe hung | ✅ Remove SD → boot eMMC → hot-insert SD → fix the service (`Type=simple`, `timeout` per step, log file) → reboot |
| SD system mounts the eMMC | `EXT4-fs (mmcblk0p2): mounted ... r/w` | TI image automount | ⚠️ `umount /dev/mmcblk0p2` before pulling power |
| Runtime `unbind`/`bind` of `k3_r5_rproc` | Kernel oops (`unbind_store`), driver broken until reboot | Driver bug when core1 never initialised | ❌ Don't. Use `rproc-late` at boot |
| `mcu3_0` firmware load `error -22` | Error in `dmesg` | TI ships `mcu3_0/mcu3_1` as **0-byte files** (unused) | ✅ Harmless |
| `cma: Failed to reserve 512 MiB` | Boot message | Only ~2.1 GB RAM visible to Linux | ⚠️ Harmless so far; drop `cma=512M` if memory problems appear |
| Can't `fsck` the SD rootfs from eMMC | `unsupported feature(s): FEATURE_C12` | ext4 `orphan_file`; Debian 11's `e2fsck` too old | ⚠️ Rely on the kernel's journal replay on mount. The FAT partition checks fine with `fsck.vfat -a` |
| Logs from a stuck boot are gone | `/var/log` empty | TI image: `/var/log -> volatile/log` (RAM) | ✅ Write your own logs under `/root` |
| 8.2-only options on SDK 10 | `platform`/`version`/`deny_list`/`ti_internal_nc_flag` rejected | Option names changed in TIDL 10 | ✅ Compile and run scripts switch on the onnxruntime version (1.7 = 8.2). TIDL 10: `deny_list:layer_type`, `ORT_DISABLE_ALL` at compile **and** inference |
| TIDL 10 output shapes | `{1,1,1,C,H,W}` instead of `{1,C,H,W}` (ORT `VerifyOutputSizes` warnings) | TIDL 10 output layout | ✅ Reshape outputs to the declared ONNX shapes (the warnings stay; harmless) |
| Full YOLO26 with attention on the NPU fails | `TIVX_CMD_NODE_CREATE failed for node TIDLNode`, `Verify OpenVX graph failed`, `TIDL Compute Invoke Failed` on the first run | **Softmax** (next row). Found with the C7x log at `debug_level 3` and micro-models | ✅ Softmax on the ARM, everything else on the NPU (split pipeline, §8) |
| Can't stop the C7x at runtime | `echo stop > .../state`: `Connection timed out`; `can't stop rproc: -110` | Vision-apps firmware doesn't acknowledge stop | ⚠️ `poweroff`, or drop the DSP modprobe from the rproc-late script and reboot |
| **Softmax can't run on this C7x** | C7x log: `Output Transpose is not supported on this device..` → `WorkloadUnitExec_Init ... Failed` → `ialg.algInit failed with status = 1` | TIDL 10.0's SoftMax kernel uses an "output transpose" the TDA4VM C7x (v1) lacks. Every form tried fails (4-D/3-D, after MatMul or not). **The PC compiler and emulator accept it and give correct numbers**: only the board shows it | ⚠️ Force Softmax onto the ARM: 2 one-node CPU stages (0.3 ms each). Transpose, MatMul and q@kᵀ all run fine on the NPU |
| Original attention gives garbage on TIDL 10 | Emulated attention block: mean error 0.57 on outputs ≤ 2.8 | TIDL 10 mis-compiles the Ultralytics layout (Split on axis 2, `qᵀk`, `v @ attnᵀ`) | ✅ Export attention in DeiT form (separate q/k/v convs, DeiT order, same math): mean error 0.04 |
| TIDL's own deny-list partitioning gives wrong numbers | DeiT block with Softmax denied: 2 subgraphs, error as bad as the original layout | Hand-off between TIDL subgraphs scrambled | ✅ Use the explicit split pipeline instead; each half alone is correct |
| TIDL 10 import fails with no message | `ERROR: - Failed in function: tidl_optimizeNet` | A part output that comes straight from a Split/Reshape/Transpose | ✅ Sink Reshape/Transpose into the consuming part and add an identity depthwise-conv leaf after such outputs |
| NPU wedged after a "slow" run | The NPU health-check model no longer answers | **Two TIDL processes at once** (a leftover run + a new one) | ✅ Reboot (warm is fine, §2.2). Run one TIDL process at a time; check with `pgrep` first |
| `pgrep -f`/`pkill -f` misbehave over ssh | Wait loop never ends; `pkill` kills the ssh session (exit 255) | The pattern matches the remote shell's own command line | ✅ Use `pgrep -x python3`, or a pattern not contained in the command |
| Accuracy looks terrible in op tests | Large "error" on Conv and Sigmoid outputs | Test fed random 0–255 noise; ranges calibrated on real frames saturate | ✅ Test on real letterboxed images. Board output is **bit-identical** to PC emulation |

---

## 8. NPU work on the board: commands

### eMMC Debian (SDK 8.2)
```bash
ssh debian@192.168.7.2
cd <workdir>
echo <pw> | sudo -S -E env PYTHONNOUSERSITE=1 timeout 60 python3 <zoo_smoke_test>.py   # sanity: ~11 ms
echo <pw> | sudo -S -E env PYTHONNOUSERSITE=1 timeout 300 python3 run_npu.py <model_dir> [--cpu]
```

### SD system (SDK 10)
```bash
ssh root@<board-eth-ip>
cd <workdir>
timeout 60  python3 op_test.py <micro_model_dir> <image>.png     # single-op check, NPU vs CPU
timeout 400 python3 run_npu.py <model_dir> --out <results_dir>   # full model on a folder of images (ONE TIDL process at a time)
cat /root/<rproc-late log>                                       # remote-core bring-up log
```

### Compile (PC)
```bash
# SDK 8.2 artifacts
MSYS_NO_PATHCONV=1 docker run --rm -v "<project>:/work" <tidl-8.2-image> python3 compile_tidl.py <onnx|split_dir> --bits 8
# SDK 10, recommended model (everything on the NPU except Softmax)
<python> export_tidl_onnx.py <checkpoint>.pt --branch one2one --imgsz 320 --attn deit
<python> split_tidl.py <onnx> --cpu-ops Softmax --out <split_dir>
MSYS_NO_PATHCONV=1 docker run --rm -v "<project>:/work" <tidl-10-image> python3 compile_tidl.py <split_dir> --bits 16 --out <model_dir>
# small-block / op tests: extract_block.py (one attention block + real calib tensors), make_micro.py (single-op models)
MSYS_NO_PATHCONV=1 docker run --rm -v "<project>:/work" <tidl-10-image> sh -c "python3 compile_tidl.py <dir>/model.onnx --feeds <dir>/calib.npz --out <dir> --deny ''; python3 op_test.py <dir> <dir>/input.npy"
# bit-exact target emulation on the PC (same artifacts)
MSYS_NO_PATHCONV=1 docker run --rm -v "<project>:/work" <tidl-10-image> python3 op_test.py <dir> <image> out.npy
```

### Deploy
Copy the compiled model folder **without `tempDir`**, plus the runtime scripts (`pipeline.py`, `run_npu.py`, `obb_common.py`, `op_test.py`), e.g. by streaming a tar archive over SSH:
```bash
(cd <model_dir> && tar cf - --exclude=tempDir .) | ssh <user>@<board> "mkdir -p <workdir>/<name> && tar xf - -C <workdir>/<name>"
```
Use `tar xm` on the SD system to avoid the clock-skew warnings.

### Debug tools
- **C7x/remote-core log:**
  - eMMC: `sudo sh -c "timeout 80 /opt/vision_apps/vx_app_arm_remote_log.out > remote.log &"`
  - SD: `timeout 40 /opt/vx_app_arm_remote_log.out`
  - Then run the model. With `debug_level 3` you get per-layer traces (don't time with it: it also causes bogus ENOMEM errors). A healthy run shows `TIDL_process is started` and a per-layer profile table.
  - On SDK 10 the actual reason for a `TIDL_create` failure (e.g. the Softmax problem) is printed **only at `debug_level 3`** (`TIDL_DEBUG=3 python3 op_test.py ...`). The log reader first dumps boot history; filter by uptime.
- **Remote firmware traces:** `mount -t debugfs none /sys/kernel/debug; cat /sys/kernel/debug/remoteproc/remoteprocN/trace0`. Empty traces mean the firmware stalled before its first print (wrong SYSFW, §7).
- **Test new ops small first.** Extract a tiny model (`onnx.utils.extract_model`), compile it, and run it with `timeout`. A bad op can wedge the C7x.
- **Is the board sending anything?** `Get-NetAdapterStatistics -Name Ethernet` twice, 30 s apart (PowerShell, no admin).

---

## 9. Results

| Test (8-bit) | SDK 8.2 (eMMC) | SDK 10.0 (SD, BOOT held) |
|---|---|---|
| TI zoo RegNetX-200MF | ✅ 3.6 ms | ✅ 2.9 ms |
| Cast→Conv | ✅ 5.2 ms | ✅ 2.4 ms |
| Cast→Conv→Sigmoid | ❌ C7x hang | ✅ 2.5 ms, mean err 0.026 (of 1.0) |
| Cast→Conv→SiLU | ❌ C7x hang | ✅ 2.6 ms |
| Board vs PC emulation | n/a | bit-identical |
| YOLO26n-OBB, ARM only | 208 ms/frame, exact | n/a |
| YOLO26n-OBB as one TIDL graph incl. Softmax | ❌ | ❌ `TIDL_create` fails: Softmax unsupported on this C7x |
| Micro: Transpose / MatMul / q@kᵀ / Softmax | n/a | ✅ / ✅ / ✅ / ❌ |

YOLO26n-OBB on SDK 10, measured on a small set of 16 test frames against a float ONNX run on the PC with the same preprocessing:

| Pipeline | Inference | Class match | Texts identical |
|---|---|---|---|
| ARM only (SDK 8.2) | 208 ms | 100% | 16/16 |
| Whole attention on ARM, 8-bit | 17.4 ms | 95.0% | 4/16 |
| DeiT attention, only Softmax on ARM, 8-bit | 16.9 ms | 94.0% | 7/16 |
| Whole attention on ARM, 16-bit | 19.0 ms | 98.7% | 12/16 |
| **DeiT attention, only Softmax on ARM, 16-bit** (recommended) | **18.7 ms** | **98.7%** | **12/16** |

- Plus ~5 ms preprocessing and ~3 ms postprocessing on the ARM.
- Some remaining "mismatches" are the float reference misreading (e.g. an extra digit). The test frames were also the calibration set, so all numbers are optimistic.

---

## 10. Checklist

1. Before any NPU run: `timeout` around it, a clean board (reboot after any hang), and on SD check that `dmesg | grep "firmware rev"` shows 11.1.8.
2. Power-cycling means **both** cables out, LEDs dark.
3. SD system: BOOT held at power-on. eMMC system: SD card removed.
4. Never add a blocking boot service. Use `Type=simple` plus logs under `/root`.
5. Long board jobs: detach them (`nohup … > file &`) and poll.
6. Don't trust accuracy numbers from random input; use real letterboxed frames.
7. `docker run` from Git Bash: `MSYS_NO_PATHCONV=1`.
8. Get a 3.3 V USB-UART cable before any bootloader experiment.
9. Only one TIDL process on the board at a time.
10. "Compiles and emulates fine on the PC" proves nothing on this C7x: test new ops on the board with a micro-model first.

---

## 11. How it works (the big picture)

### 11.1 The pieces
```
 PC (Windows)                                   Board (SD system, TI SDK 10)
 ───────────────────────────────────────────    ─────────────────────────────────────────
 checkpoint .pt                                 compiled model folder (copied over)
   │ export_tidl_onnx.py   (conda env)            │
   ▼                                              │ run_npu.py / pipeline.py
 <name>.onnx  (+ .json sidecar)                   │   onnxruntime + TIDLExecutionProvider
   │ split_tidl.py         (conda env)            │     s00_tidl ──► C7x/MMA (NPU)
   ▼                                              │     s01_cpu  ──► ARM A72 (Softmax)
 split folder (s00_tidl.onnx, s01_cpu…)           │     s02_tidl ──► NPU
   │ compile_tidl.py       (Docker, TIDL 10)      │     s03_cpu  ──► ARM
   ▼                                              │     s04_tidl ──► NPU
 compiled folder (ONNX + NPU .bin files) ─────────┘   obb_common.py: decode + top-k + text (ARM)
```

### 11.2 What runs where, and why
| Part | Where | Why |
|---|---|---|
| Letterbox to 320×320 | ARM (numpy/cv2) | Plain image resize |
| Backbone, neck, heads: Conv, SiLU (Sigmoid·Mul), Add, Concat, MaxPool, Resize, Split | **NPU** | All supported by TIDL 10 |
| Attention: q/k/v convs, Reshape, Transpose, MatMul (q@kᵀ and attn@v), pe conv, proj | **NPU** | Supported on this C7x once exported in DeiT form |
| Attention Softmax (2×100×100, twice) | ARM | TIDL 10 Softmax needs an "output transpose" this C7x lacks. Costs ~0.3 ms each |
| Box decode, sigmoid of class scores, top-k, angle regularisation, text reading | ARM (numpy) | TopK/GatherElements/Sin/Cos aren't NPU ops; tiny work anyway |

### 11.3 What "the model format" is
The board does not run a plain ONNX file on the NPU. Each NPU part is **two things that must stay together**:

- `sNN_tidl.onnx`: the ONNX graph of that part. onnxruntime reads it to know the inputs, the outputs and which nodes to offload.
- `sNN_tidl/artifacts/`: TI's compiled form, made by the TIDL compiler on the PC:
  - `subgraph_0_tidl_net.bin`: the network compiled for the C7x/MMA (fused layers, quantised weights, memory plan). **This is what the NPU executes.** The ONNX weights are not used for offloaded nodes.
  - `subgraph_0_tidl_io_1.bin`: input/output buffer layout and quantisation scales.
  - `allowedNode.txt`, `onnxrtMetaData.txt`: the node list and metadata onnxruntime needs to match the ONNX to the `.bin`.
  - `tempDir/`: compiler debug output (layer list `*_netLog.txt`, graphs). Not needed on the board; leave it out when copying.

The `.bin` files are tied to **the TIDL version** (8.2 and 10.0 artifacts are not interchangeable), **the chip** (`am68pa` = TDA4VM) and **the exact ONNX** they were compiled from. Never edit or swap one without recompiling.

`pipeline.json` (a project-specific format written by `split_tidl.py`) lists the stages in order: name, `tidl`/`cpu`, model file, input tensor names, and output names (`[name inside the part, global name]`). `model.json` holds class names, strides, image size and branch.

There is also a lower-level route without onnxruntime (TI's TIDL-RT C API loading `*_tidl_net.bin` + `*_tidl_io_*.bin` directly). It isn't used here.

### 11.4 The model itself
- **YOLO26n-OBB**, 14 classes: `0`–`9`, `:`, `M`, `_`, `line2` (a box around the second text line). Input 320×320. Reads dot-matrix date/lot codes.
- Two heads exist: `one2many` (needs NMS) and `one2one` (NMS-free, top-k). **Use `one2one`** from a checkpoint whose one2one head is trained properly (one checkpoint variant had a broken one2one head, §7).
- The ONNX stops after the head convolutions and returns 9 raw tensors: `box_s8, cls_s8, ang_s8, box_s16, cls_s16, ang_s16, box_s32, cls_s32, ang_s32` (shapes `[1,4,H,W]`, `[1,14,H,W]`, `[1,1,H,W]` with H=W=40, 20, 10).

---

## 12. Quick start: run a finished model

The board must be booted into the SD system (§2.2).

```bash
# 1. Board alive and on the right firmware? (expect 11.1.8 and 11)
ssh root@<board-eth-ip> 'dmesg | grep "firmware rev"; ls /sys/bus/rpmsg/devices | grep -c chrdev'

# 2. Copy the compiled model, a tiny health-check model, the scripts and the test images
#    (leave out tempDir and calibration .npz files)
tar cf - --exclude=tempDir --exclude='*_calib.npz' <model_dir> <health_model_dir> <scripts> <images> \
  | ssh root@<board-eth-ip> 'mkdir -p <workdir> && tar xmf - -C <workdir>'
```

Then on the board:

```bash
timeout 40  python3 op_test.py <health_model_dir>                 # NPU health (e.g. a Conv→Sigmoid micro-model): PASS, ~2.5 ms
timeout 400 python3 run_npu.py <model_dir> --out <results_dir>
#   per image: number of detections and the decoded text; at the end: median ms per stage (infer ≈ 19 ms)
#   writes <results_dir>/results.json and an annotated PNG per image
```

`run_npu.py` options: `--images "glob"`, `--cpu` (same model on the ARM only), `--conf 0.25`, `--warmup 5`, `--repeat 10` (timed runs per image), `--out DIR`.

Back on the PC, copy the results folder back and score it against a float reference:
```bash
<python> compare_results.py <results_dir>/results.json --baseline <float_reference>/results.json
#   prints: detections in reference vs run, matched, same class (%), max center error (px),
#   number of images with identical text, and every differing text (ref vs run)
```

Rules: one TIDL process at a time; always use `timeout`; after any hang run `sync; reboot` (a warm reboot keeps the right firmware).

---

## 13. Setting up a PC from scratch

### 13.1 Python (conversion, splitting, scoring)
```bash
conda create -n <env> python=3.10 -y
<python> -m pip install torch --index-url https://download.pytorch.org/whl/cpu
<python> -m pip install ultralytics==8.4.166 onnx onnxsim onnxruntime opencv-python numpy paramiko
```
- Keep `ultralytics` at the version the export was validated with (8.4.166). A newer one can change `Attention`, `OBB26` or `dist2rbox`; then re-check the export and the decode (§14.5).
- Call the env's `python.exe` directly; `conda run` breaks multi-line `-c`.

### 13.2 TIDL compilers (Docker)
Start Docker Desktop and wait until `docker info` answers. Then build one image per TIDL version:
- **TIDL 10.0** (for the SD system): Ubuntu 22.04, python 3.10, `edgeai-tidl-tools` 10_00_02_00 for SOC `am68pa`, onnxruntime-tidl `1.14.0+10000005` (same build as the board). ~0.9 GB.
- **TIDL 8.2** (only for the eMMC system): Ubuntu 18.04, python 3.6, tools 08_02_00_01, onnxruntime-tidl 1.7.0. ~0.7 GB.
- Both mount the project at `/work`. From Git Bash always prefix `MSYS_NO_PATHCONV=1`, or Docker gets Windows-mangled paths.
- The compiler version must match the board's TIDL exactly, or the artifacts won't load.

### 13.3 SSH
Generate a key if there is none (`ssh-keygen -t ed25519`). Put the `.pub` line into the board's `/root/.ssh/authorized_keys` (SD system) or the `debian` user's `~/.ssh/authorized_keys` (eMMC). On the first connection to a new IP, add `-o StrictHostKeyChecking=accept-new`.

---

## 14. Converting a model, step by step

The example is a retrained checkpoint with the same YOLO26n-OBB architecture.

### 14.1 Export to a TIDL-friendly ONNX
```bash
<python> export_tidl_onnx.py <checkpoint>.pt --branch one2one --imgsz 320 --attn deit
#   -> <name>_one2one_320_deit.onnx + a .json sidecar
```
What the export changes compared with a stock `yolo export` (none of it changes the math):

| Change | Why |
|---|---|
| Conv+BN fused; `/255` folded into the first conv | Input is raw 0–255 pixels; no extra op |
| Graph cut after the head convs → 9 raw outputs | TopK/GatherElements/Sin/Cos aren't NPU ops; each output gets its own quantisation scale |
| SPPF MaxPool 5×5 → two 3×3 | 5×5/s1 unsupported by TIDL 8.2 (harmless for 10) |
| Attention `q*scale` folded into the q conv weights | One op fewer |
| DeiT attention: separate q/k/v 1×1 convs, `qᵀ@k → softmax → attn@vᵀ`, pe fed from the v conv | TIDL 10 mis-compiles the original layout. A per-head unrolled variant and the original Ultralytics layout are also available |
| Short tensor names (`t123_0`, `w45`) | Long names crash the TIDL importer |
| opset 11 (configurable), IR ≤ 7, onnxsim | Board onnxruntime compatibility |

Options: `--branch one2one|one2many` (required), `--imgsz 320`, `--attn orig|deit|deit_unroll`, `--opset 11`, `--out path`.

**Check it.** The float ONNX must equal PyTorch: register the ONNX/checkpoint pair in `validate_onnx_pc.py` and run it. Expect all PyTorch detections matched with a max center error of 0.000 px (a couple of score ties at top-k are fine).

### 14.2 Split into NPU / ARM parts
```bash
<python> split_tidl.py <onnx> --cpu-ops Softmax --out <split_dir>
#   lists each stage (s00_tidl: N nodes, s01_cpu: 1 node, ops Softmax, ...), then
#   pipeline vs original, max abs diff: 0.00e+00      <- must be 0
```
- `--cpu-ops`: op types forced onto the ARM. Everything else becomes NPU "islands". `Softmax` is the only one needed on SDK 10. The default `MatMul,Softmax,Transpose,Reshape,Split` is the SDK 8.2 setting: the whole attention on the ARM.
- `--min-tidl 8`: NPU islands smaller than this run on the ARM instead.
- The image input becomes **uint8 + Cast**. This is lossless and matches TI's zoo models.
- It works around two importer quirks automatically: no NPU part may end on a Reshape/Transpose/Split output, and every NPU part output becomes a leaf (an identity 3×3 depthwise conv copy if the tensor is also used inside the part).
- **Why not let TIDL partition with a deny list?** Its own partitioning is order-sensitive and gave wrong numbers at the subgraph hand-off. The explicit split compiles every NPU part as a standalone model.

### 14.3 Compile for the NPU (Docker, slow)
```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "<project>:/work" <tidl-10-image> \
  python3 compile_tidl.py <split_dir> --bits 16 --out <model_dir> > compile.log 2>&1
grep -aE "== compiling|Final number|pipeline compiled|produced no" compile.log
```
- It takes ~10–20 min (16 calibration images × 5 iterations per part). Run it in the background.
- **Calibration**: the float pipeline is run on the calibration images (`--calib "<glob>"`, `--calib-iters 5`), and each NPU part is calibrated on its real input tensors. Use images that look like production data; more variety gives better ranges.
- `--bits 16` is recommended: 98.7% class match vs 94% at 8-bit, for +2 ms. `--bits 8` is the alternative.
- Other options: `--qscale 0|1` (power-of-2 scales), `--data-convert 3` (keep it), `--deny` (single-ONNX mode only).
- Harmless noise in the log: `VX_ZONE_ERROR:Enabled`, `TIDL_E_DATAFLOW_INFO_NULL` (first pass), a segfault/`free(): invalid pointer` at the very end of each part (the artifacts are already written).
- A real failure prints `compile of sNN_tidl produced no artifacts`. Look for `[TIDL Import] ERROR` in the log (§16.3).
- For a quick diagnostic compile, use one or two calibration images and `--calib-iters 1`.

### 14.4 Deploy and run
Follow §12 with the new model folder. To score it, make a float reference with the **same preprocessing** (PC, no NPU):
```bash
<python> run_npu.py <split_dir> --cpu --out <float_reference>       # any pipeline folder works with --cpu
<python> compare_results.py <board_results>/results.json --baseline <float_reference>/results.json
```
Don't use a PyTorch baseline for this if it used a different letterbox.

### 14.5 If the architecture or Ultralytics version changes
- Re-check that `Attention` still has `qkv`, `pe`, `proj`, `num_heads`, `key_dim`, `head_dim` and `scale` (the export's scale-folding and DeiT-conversion code depends on them).
- Re-check the head: `OBB26` with `cv2` (box), `cv3` (class), `cv4` (angle) and the `one2one_*` copies; reg_max = 1 (no DFL).
  - The numpy decode mirrors `OBB26` + `dist2rbox`.
  - The center offset `(rb−lt)/2` is rotated by the **raw** angle, then the anchor (0.5) is added and everything is multiplied by the stride; `wh = lt+rb`.
  - Then w ≥ h and angle ∈ [0, π/2) are enforced.
- The ONNX-vs-PyTorch validation must still give 0.000 px. If not, fix the export or decode before going further.
- After any graph change, test new op types on the board first (§16.2).

---

## 15. Using the model from your own code (on the board)

Input contract (`obb_common.letterbox`):
1. Read the image with OpenCV (BGR).
2. Resize, keeping the aspect ratio, to fit 320.
3. Pad to 320×320 with value **114**, centred.
4. Convert **BGR→RGB**.
5. Reorder to NCHW, values 0–255, **no normalisation**. The pipeline casts to uint8 itself.

```python
import json, cv2
from obb_common import letterbox, postprocess, to_image_coords, read_text, draw
from pipeline import Pipeline, tidl_session_factory

d = "<model_dir>"
meta = json.load(open(d + "/model.json"))
names = {int(k): v for k, v in meta["names"].items()}
model = Pipeline(d, tidl_session_factory(d))    # opens all stages once (~1 s). Keep it; one per process.

img = cv2.imread("<image>.png")
x, ratio, pad = letterbox(img, meta["imgsz"])   # (1,3,320,320) float32 RGB 0..255
outs = model.run(x)                             # 9 arrays, in meta["outputs"] order
dets = to_image_coords(postprocess(outs, meta["branch"], conf=0.25, strides=meta["strides"]), ratio, pad)
# dets: (K,7) = x_center, y_center, w, h, angle_rad, score, class   in original image pixels
print(read_text(dets, names))                   # text lines top->bottom, "line2" skipped
cv2.imwrite("out.png", draw(img, dets, names))
```
- Run as root on the SD system (no `sudo` needed). Wrap the process in `timeout` while developing.
- `postprocess` for `one2one`: top-300 by score, keep `score > conf`. For `one2many`: class-aware rotated NMS (`iou=0.7`).
- `model.times` holds the last per-stage times in seconds.
- To run on the CPU only (PC or board), use `Pipeline(d)` with no factory.
- Keep `obb_common.py` and `pipeline.py` py3.6-compatible: they also run inside the 8.2 compiler container.

---

## 16. Accuracy, new ops, debugging

### 16.1 Accuracy is worse than expected
1. Compare against a float run with the same preprocessing (§14.4), not PyTorch.
2. Look at the annotated PNGs: are the misses on hard frames (smeared print) or everywhere?
3. Switch 8-bit → 16-bit (`--bits 16`). This is the biggest single win.
4. Improve calibration: more and more varied images, `--calib-iters 10`.
5. Check whether the loss is on the NPU or the ARM side: run the same model folder with `--cpu`. If that matches the float reference, the loss is quantisation.
6. Find the worst part: run each NPU part alone with `op_test.py` on real inputs (§16.2) and compare errors.
7. Not tried yet: mixed precision with `advanced_options:output_feature_16bit_names_list` (16-bit only for sensitive layers, e.g. the heads), or `accuracy_level`.

### 16.2 Testing a new op or block on the NPU (before compiling a big model)
"Compiles and emulates fine on the PC" proves nothing for this C7x (the Softmax case). Test on the board with a tiny model:

- **One attention block of a real model**, with real calibration tensors: `extract_block.py <onnx> --block 0 --out <dir>` writes `model.onnx`, `calib.npz` and `input.npy`.
- **Single-op micro models**: `make_micro.py` writes one small model per pattern (each on a `[1,64,10,10]` input); add new patterns in its builder.
- Compile and emulate on the PC:
  ```bash
  MSYS_NO_PATHCONV=1 docker run --rm -v "<project>:/work" <tidl-10-image> sh -c \
    "python3 compile_tidl.py <dir>/model.onnx --feeds <dir>/calib.npz --out <dir> --deny ''; \
     python3 op_test.py <dir> <dir>/input.npy"
  ```
- Copy the folder (without `tempDir` and `calib.npz`) to the board and run `timeout 60 python3 op_test.py <dir> <dir>/input.npy`.

`op_test.py <dir> [input]` runs `<dir>/model.onnx` with `<dir>/artifacts` two ways and compares them:
- On the NPU (or in TI's bit-exact emulation inside the container), and on the CPU.
- It prints the time and `max/mean/p99 |diff|` against the float result.
- The input can be an image path (letterboxed like the model), a `.npy` (one input), an `.npz` (several named inputs), or nothing (random 0–255; only meaningful for image inputs, see §7).
- On a healthy board the result equals the emulation exactly.

Known op status on this board:
- **TIDL 10.0:** Conv, Sigmoid/SiLU, Add/Mul, Concat, MaxPool, Resize, Split (channel), Reshape, Transpose, MatMul ✅ · **Softmax ❌** (create fails).
- **TIDL 8.2:** Conv ✅ · Sigmoid ❌ (hang), MatMul ❌, HardSigmoid/Tanh not offloaded.

### 16.3 Reading failures
| Where | Message | Meaning / next step |
|---|---|---|
| Compile | `[TIDL Import] ERROR: - Failed in function: tidl_optimizeNet` | The importer rejected the graph without a reason. Usually a part ending on a data-movement op. Compile with `-e TIDL_DEBUG=2` (after `docker run --rm`) and look at the output-tensor list just before the error |
| Compile | `produced no artifacts` | The part failed; search the log for the first `ERROR` |
| Board | `TIVX_CMD_NODE_CREATE failed for node TIDLNode` / `Verify OpenVX graph failed` / `TIDL Compute Invoke Failed` (fast) | The C7x refused the network at create. To get the reason: start `timeout 40 /opt/vx_app_arm_remote_log.out > c7x.log &`, run with `TIDL_DEBUG=3` (`op_test.py` honours it), then `grep C7x c7x.log` for lines after the current uptime (e.g. `Output Transpose is not supported`) |
| Board | No output; `timeout` kills it (exit 124) | C7x hang. The health-check model will now hang too → `sync; reboot`. Then bisect with smaller models |
| Board | `MEM: Alloc failed status=12` | Usually a wedged C7x or `debug_level 3` → reboot |
| Board | `IPC: Unable to create TX channels` | Wrong boot (SYSFW 21.9.1) or remote cores not up → check §2.2 |
| Board | `VerifyOutputSizes ... {1,1,1,C,H,W}` warnings | Harmless (TIDL 10 output layout) |

### 16.4 Tools
| Script | Runs on | Purpose |
|---|---|---|
| `export_tidl_onnx.py` | PC | `.pt` → TIDL-friendly ONNX + `.json` sidecar (§14.1) |
| `validate_onnx_pc.py` | PC | ONNX + numpy postprocess vs PyTorch |
| `split_tidl.py` | PC | ONNX → NPU/ARM parts + `pipeline.json`, with self-check (§14.2) |
| `compile_tidl.py` | Docker | Split folder, single ONNX, or one part (`--feeds`) → TIDL artifacts (§14.3) |
| `pipeline.py` | everywhere | Runs a split pipeline (NPU or CPU sessions) |
| `obb_common.py` | everywhere | letterbox, decode, postprocess, text, draw |
| `run_npu.py` | board / PC | Full test run on a folder of images → `results.json` + PNGs |
| `compare_results.py` | PC | Score a run against a reference |
| `op_test.py` | board / Docker | One small model: NPU (or emulation) vs CPU |
| `extract_block.py` | PC | Cut one attention block + real calib tensors |
| `make_micro.py` | PC | Single-op test models |
| `baseline_pt.py` | PC | PyTorch reference (uses a different letterbox) |

Calibration and test data: 900×600 frames of printed date/lot codes. When the same frames serve as both calibration and test set, accuracy on them is optimistic.
