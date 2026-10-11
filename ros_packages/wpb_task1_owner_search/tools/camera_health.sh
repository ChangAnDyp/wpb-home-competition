#!/usr/bin/env bash
# 相机体检 / 一键恢复。现场排查"相机掉线"先跑这个，别靠猜。
#
#   bash camera_health.sh            # 只体检，不动任何东西
#   bash camera_health.sh --recover  # 体检 + 按正确顺序做一次恢复
#
# 恢复顺序（很重要，顺序反了会把还能救的相机弄死）：
#   ① 杀 bridge nodelet ──→ 等 respawn
#   ② 还不行 → 杀整个 nodelet manager（真正持设备的进程）
#   ③ 还不行 → 确认没人持设备后，sysfs authorized 软拔插（需要 sudo NOPASSWD，
#      见 install_camera_usb_permission.sh；没装就退化成 USBDEVFS_RESET）
# 说明：本脚本对应看门狗里自动跑的同一条梯子；跑这个是为了"人在现场"时看得清。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TOPIC="${IMAGE_TOPIC:-/kinect2/qhd/image_color_rect}"
KEEPALIVE_TOPIC="${KEEPALIVE_TOPIC:-/kinect2/sd/camera_info}"
BRIDGE_NODE="${BRIDGE_NODE:-/kinect2_bridge}"
MANAGER_NODE="${MANAGER_NODE:-/kinect2}"
POWER_HELPER="${POWER_HELPER:-/usr/local/sbin/kinect2-usb-powercycle}"

source /opt/ros/noetic/setup.bash 2>/dev/null || true
source "$HOME/catkin_ws/devel/setup.bash" 2>/dev/null || true

hr() { printf '%s\n' "------------------------------------------------------------"; }

section() { hr; echo "== $*"; }

section "基础状态"
date
uptime
echo "ROS_MASTER_URI=${ROS_MASTER_URI:-<未设置>}"
echo "image_topic=$IMAGE_TOPIC"

section "关键节点是否在跑"
rospy_nodes=$(rosnode list 2>&1)
if [[ "$rospy_nodes" == *"ERROR"* || -z "$rospy_nodes" ]]; then
  echo "rosnode list 失败：$rospy_nodes"
  echo "→ 栈没起来？先 robot.sh stop，再重新起栈"
else
  echo "$rospy_nodes" | grep -E "kinect2|watchdog|tts|asr|task_all_owners" || echo "（没有相机/语音节点）"
fi

section "相机话题有没有发布者"
rostopic list 2>/dev/null | grep -E "^/kinect2/qhd|^/kinect2/sd/camera_info" || echo "（话题都不在：bridge 没起来）"

section "画面帧率（测 5 秒，正常 13~15Hz）"
timeout 6 rostopic hz "$IMAGE_TOPIC" 2>&1 | tail -3 || echo "（拿不到帧率 → 掉流/没发布者）"

section "订阅者数量（bridge 在'没有订阅者'时会自己 stop 设备）"
rostopic info "$IMAGE_TOPIC" 2>&1 | sed -n '/Subscribers/,$p' || true
rostopic info "$KEEPALIVE_TOPIC" 2>&1 | sed -n '/Subscribers/,$p' || true

section "USB 层：设备在不在、有没有被复位过"
lsusb 2>/dev/null | grep -i "045e:02c4" || echo "（lsusb 里没有 045e:02c4！设备整机掉线或没上电）"
for d in /sys/bus/usb/devices/*/; do
  if [[ "$(cat "${d}idVendor" 2>/dev/null)" == "045e" && "$(cat "${d}idProduct" 2>/dev/null)" == "02c4" ]]; then
    echo "sysfs: ${d}  speed=$(cat "${d}speed" 2>/dev/null)  authorized=$(cat "${d}authorized" 2>/dev/null)  devnum=$(cat "${d}devnum" 2>/dev/null)"
  fi
done
dmesg 2>/dev/null | grep -E "4-2" | tail -8

section "结论"
if timeout 6 rostopic hz "$IMAGE_TOPIC" 2>/dev/null | grep -q "average rate"; then
  echo "相机有画面：可以先跑任务。"
else
  echo "相机没有画面 → 需要恢复。"
  if [[ "${1:-}" != "--recover" ]]; then
    echo "（加 --recover 让本脚本按正确顺序做一次恢复）"
    exit 1
  fi
fi

[[ "${1:-}" == "--recover" ]] || exit 0

section "恢复 ①：重载 bridge nodelet"
rosnode kill "$BRIDGE_NODE" 2>&1 || true
for _ in $(seq 1 30); do
  sleep 1
  if timeout 5 rostopic hz "$IMAGE_TOPIC" 2>/dev/null | grep -q "average rate"; then
    echo "已恢复（重载 bridge 就够了）。"
    exit 0
  fi
done

section "恢复 ②：重启 nodelet manager（释放被卡住的相机句柄）"
rosnode kill "$MANAGER_NODE" 2>&1 || true
for _ in $(seq 1 45); do
  sleep 1
  if timeout 5 rostopic hz "$IMAGE_TOPIC" 2>/dev/null | grep -q "average rate"; then
    echo "已恢复（重启 manager 救回来了）。"
    exit 0
  fi
done

section "恢复 ③：USB 软拔插（先确认没人持设备）"
if sudo -n "$POWER_HELPER" 2>&1; then
  : # 成功，下面等画面
elif [[ -x "$POWER_HELPER" && $EUID -eq 0 ]]; then
  "$POWER_HELPER" 2>&1 || true
else
  echo "带 root 的软拔插不可用（没装 $POWER_HELPER 或没有 NOPASSWD）。"
  echo "→ 请人工断 Kinect 12V 电源 3 秒后重插，然后 robot.sh stop + 重新起栈。"
fi
for _ in $(seq 1 60); do
  sleep 1
  if timeout 5 rostopic hz "$IMAGE_TOPIC" 2>/dev/null | grep -q "average rate"; then
    echo "已恢复（USB 软拔插救回来了）。"
    exit 0
  fi
done

echo "三级恢复都没救回来 → 人工处理：断 Kinect 12V 电源重插，再 robot.sh stop + 起栈。"
exit 2
