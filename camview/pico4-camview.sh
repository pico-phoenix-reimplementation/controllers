#!/bin/sh
# Show the PICO 4 tracking cameras in VR (screens or passthrough): power the OV7251s, route
# CSIPHY5/4/3/2(proxy for CSIPHY3 combo) -> CSID0-3 -> VFE0-3 RDI0 (RAW10 640x481), start pico4-camview
# under Monado (pico4-xr), then start the Nordic MCU so the sensors stream.
#
#   sudo pico4-camview.sh                  screens, Ctrl-C to quit
#   sudo pico4-camview.sh --passthrough    black-and-white passthrough
#   PICO_MCU_EXPOSURE=400,64 CAMVIEW_GAIN=2 CAMVIEW_ROLL=0,90,0 sudo -E ...
set -eu
# The routing loop below reuses "$@"; keep the app arguments (--passthrough).
app_args="$*"
here=$(cd "$(dirname "$0")" && pwd)
mcu_dir=${PICO_MCU_DIR:-/home/user/pico}

echo 14 > /sys/module/qcom_camss/parameters/settle_cnt_override
# CSIPHY3 carries two sensors (D-PHY combo); the 2nd is described on CSIPHY2.
echo 3 > /sys/module/qcom_camss/parameters/csiphy_combo
echo 2 > /sys/module/qcom_camss/parameters/csiphy_combo_proxy
modprobe pico-ov7251-mcu
D=/sys/bus/platform/drivers/pico-ov7251-mcu
# Power-cycle sensors and MCU in stock order (AGENTS.md §O).
for n in 0 1 2 3; do echo camera-tracking-$n > $D/unbind 2>/dev/null || true; done
echo 0 > /sys/devices/platform/mcu-vdd/state; sleep 0.1
echo 1 > /sys/devices/platform/mcu-vdd/state; sleep 1.2
for n in 0 1 2 3; do echo camera-tracking-$n > $D/bind; done; sleep 1

M="media-ctl -d /dev/media0"
F=Y10_1X10/640x481
$M -r
i=0
for pair in "5 tracking-0" "4 tracking-3" "3 tracking-1" "2 tracking-2"; do
	set -- $pair
	$M -l "\"msm_csiphy$1\":1->\"msm_csid$i\":0[1],\"msm_csid$i\":1->\"msm_vfe${i}_rdi0\":0[1]"
	$M -V "\"ov7251-mcu camera-$2\":0[fmt:$F],\"msm_csiphy$1\":0[fmt:$F],\"msm_csid$i\":0[fmt:$F],\"msm_vfe${i}_rdi0\":0[fmt:$F]"
	i=$((i + 1))
done

systemctl stop pico-mcu 2>/dev/null || true
systemctl reset-failed pico-mcu 2>/dev/null || true
(
	sleep 4
	systemd-run --unit pico-mcu -E PICO_MCU_NO_CYCLE=1 \
		${PICO_MCU_EXPOSURE:+-E PICO_MCU_EXPOSURE=$PICO_MCU_EXPOSURE} \
		--working-directory="$mcu_dir" \
		sh -c "python3 -u pico-mcu.py start 86400 > $mcu_dir/mcu.log 2>&1" >/dev/null
) &
# The app quits cleanly on SIGINT (ends the OpenXR session); pico4-xr then
# stops monado-service. Forward signals explicitly, the shell would only see
# them after the foreground child exits.
pico4-xr "$here/pico4-camview" $app_args &
xr=$!
trap 'pkill -INT -x pico4-camview || true' INT TERM
wait $xr || wait $xr || true
systemctl stop pico-mcu 2>/dev/null || true
