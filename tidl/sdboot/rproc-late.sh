#!/bin/sh
# BBAI64 + TI SDK 10: boot the main R5Fs, then the C66x/C7x, after rootfs is up (Beagle U-Boot does not pre-boot them).
# Runs in the background (rproc-late.service, Type=simple) and logs every step, so a hang can never block boot.
L=/root/rproc-late.log
exec >>"$L" 2>&1
echo "=== boot at uptime $(cut -d' ' -f1 /proc/uptime)"
timeout 40 modprobe ti_k3_r5_remoteproc; echo "r5 modprobe rc=$? at $(cut -d' ' -f1 /proc/uptime)"; sync
sleep 3
timeout 40 modprobe ti_k3_dsp_remoteproc; echo "dsp modprobe rc=$? at $(cut -d' ' -f1 /proc/uptime)"; sync
sleep 5
for r in /sys/class/remoteproc/*; do echo "$(basename $r) $(cat $r/name) $(cat $r/state)"; done
echo "rpmsg chrdev endpoints: $(ls /sys/bus/rpmsg/devices/ | grep -c chrdev)"
dmesg | grep -iE "r5f|dsp|remoteproc|rpmsg|virtio" | grep -v "reserved mem" | tail -80
echo "=== done"; sync
