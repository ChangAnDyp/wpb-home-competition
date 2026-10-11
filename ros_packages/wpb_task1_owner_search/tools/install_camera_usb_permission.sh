#!/usr/bin/env bash
# 给相机看门狗装上"最后一级恢复"所需的权限。
#
# 装两样东西：
#   1. /usr/local/sbin/kinect2-usb-powercycle —— sysfs authorized 0→1 软拔插脚本
#   2. /etc/sudoers.d/kinect2-usb-powercycle —— 允许当前用户免密调用上面这个脚本
#
# 为什么要 root：/sys/bus/usb/devices/<dev>/authorized 是 0644 root:root，普通
# 用户写不了。看门狗用 `sudo -n`（非交互）调用它，失败就自动退化成
# USBDEVFS_RESET（不需要 root，但现场实测救回率低）。
#
# 用法：
#   bash install_camera_usb_permission.sh          # 安装/更新
#   bash install_camera_usb_permission.sh --remove # 卸载
#
# 注意：只给这一个脚本开 NOPASSWD，不要给整个 sudo 开；脚本本身不接受任何参数
# 之外的危险输入（VID:PID 写死）。

set -euo pipefail

HELPER_SRC="$(cd "$(dirname "$0")" && pwd)/kinect2-usb-powercycle"
HELPER_DST="/usr/local/sbin/kinect2-usb-powercycle"
SUDOERS_FILE="/etc/sudoers.d/kinect2-usb-powercycle"

if [[ "${1:-}" == "--remove" ]]; then
  sudo rm -f "$SUDOERS_FILE" "$HELPER_DST"
  echo "已卸载：$SUDOERS_FILE $HELPER_DST"
  exit 0
fi

if [[ ! -f "$HELPER_SRC" ]]; then
  echo "找不到 $HELPER_SRC" >&2
  exit 1
fi

sudo install -m 0755 -o root -g root "$HELPER_SRC" "$HELPER_DST"

TMP_SUDOERS="$(mktemp)"
trap 'rm -f "$TMP_SUDOERS"' EXIT
printf '%s ALL=(root) NOPASSWD: %s\n' "${USER:-$(id -un)}" "$HELPER_DST" > "$TMP_SUDOERS"
sudo install -m 0440 -o root -g root "$TMP_SUDOERS" "$SUDOERS_FILE"

sudo visudo -c -f "$SUDOERS_FILE"
echo
echo "安装完成，自检："
if sudo -n "$HELPER_DST" --status; then
  echo "OK：看门狗现在可以用这一级做 USB 软拔插了。"
else
  rc=$?
  if [[ $rc -eq 2 ]]; then
    echo "（脚本能跑了，只是现在相机不在 USB 总线上 —— 插上相机后可以再跑一次自检）"
  else
    echo "自检失败（rc=$rc）：sudo -n 免密调用没有生效，请检查 $SUDOERS_FILE" >&2
    exit 2
  fi
fi
