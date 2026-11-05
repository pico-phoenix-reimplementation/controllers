echo 14 > /sys/module/qcom_camss/parameters/settle_cnt_override
# CSIPHY3 D-PHY combo: the 2nd sensor is described on the CSIPHY2 proxy port.
echo 3 > /sys/module/qcom_camss/parameters/csiphy_combo 2>/dev/null
echo 2 > /sys/module/qcom_camss/parameters/csiphy_combo_proxy 2>/dev/null
modprobe pico-ov7251-mcu; sleep 2
D=/sys/bus/platform/drivers/pico-ov7251-mcu
for n in 0 1 2 3; do echo camera-tracking-$n > $D/unbind; done
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
	V=$($M -e "msm_vfe${i}_video0")
	(timeout 40 v4l2-ctl -d $V --set-fmt-video=width=640,height=481,pixelformat=Y10P --stream-mmap --stream-count=30 --stream-to=/home/user/pico/trk$i.raw > /home/user/pico/cap$i.log 2>&1 &)
	i=$((i+1))
done
sleep 1
systemctl stop pico-mcu 2>/dev/null; systemctl reset-failed pico-mcu 2>/dev/null
systemd-run --unit pico-mcu -E PICO_MCU_NO_CYCLE=1 ${PICO_MCU_EXPOSURE:+-E PICO_MCU_EXPOSURE=$PICO_MCU_EXPOSURE} --working-directory=/home/user/pico sh -c "python3 -u pico-mcu.py start 600 > /home/user/pico/mcu.log 2>&1"
sleep 12
grep -E "^rx: bb 01 (19|04)" /home/user/pico/mcu.log | uniq
grep "^MCU" /home/user/pico/mcu.log | head -8
ls -l /home/user/pico/trk*.raw
grep -E "csid|vfe" /proc/interrupts | awk '{s=0; for(i=2;i<=9;i++) s+=$i; print $NF, s}'
