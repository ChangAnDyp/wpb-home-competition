#!/usr/bin/env bash
# 启智机器人 一键设备自检
# 用途：确认底盘 / 激光雷达 / Kinect / 声卡 / GPU 是否都在线，可随时重跑。
# 运行：bash ~/competition/robot_selfcheck.sh
# 注意：只读取传感器数据，不会让机器人移动。

set +e
source /opt/ros/noetic/setup.bash
source "$HOME/catkin_ws/devel/setup.bash" 2>/dev/null

PORT=${ROBOT_CHECK_MASTER_PORT:-11333}
export ROS_MASTER_URI="http://127.0.0.1:${PORT}"
export ROS_HOSTNAME=127.0.0.1

echo "=========== 1. 硬件与设备节点 ==========="
echo "--- GPU ---"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv 2>&1 | head -3
echo "--- 串口设备 ---"
ls -l /dev/ftdi /dev/rplidar 2>&1
echo "--- Kinect (USB 045e:02c4) ---"
lsusb 2>&1 | grep -i "045e:02c4" || echo "未发现 Kinect！"
echo "--- 声卡 ---"
arecord -l 2>&1 | grep -E "^card" || echo "未发现录音设备！"
aplay -l 2>&1 | grep -E "^card" || echo "未发现播放设备！"

echo
echo "=========== 2. 启动 ROS 并检查传感器 ==========="
roscore -p "$PORT" > /tmp/selfcheck_roscore.log 2>&1 &
RC_PID=$!
sleep 5

rosrun wpb_home_bringup wpb_home_core _serial_port:=/dev/ftdi > /tmp/selfcheck_core.log 2>&1 &
CORE_PID=$!
rosrun rplidar_ros rplidarNode _serial_port:=/dev/rplidar _frame_id:=laser > /tmp/selfcheck_lidar.log 2>&1 &
LIDAR_PID=$!
sleep 8

echo "--- /odom (底盘，应约 100Hz) ---"
timeout 6 rostopic hz /odom 2>&1 | grep -m1 "average rate" || echo "!! /odom 无数据：检查 /dev/ftdi 与总开关"
echo "--- /scan (激光雷达，应约 12Hz) ---"
timeout 6 rostopic hz /scan 2>&1 | grep -m1 "average rate" || echo "!! /scan 无数据：检查 /dev/rplidar"

echo "--- Kinect2 彩色图 (应约 30Hz) ---"
roslaunch kinect2_bridge kinect2_bridge.launch > /tmp/selfcheck_kinect.log 2>&1 &
KIN_PID=$!
for _ in $(seq 1 12); do
  sleep 3
  RATE=$(timeout 5 rostopic hz /kinect2/qhd/image_color_rect 2>&1 | grep -m1 "average rate")
  [ -n "$RATE" ] && { echo "$RATE"; break; }
done
[ -z "$RATE" ] && echo "!! Kinect 无图像：换 USB3.0 接口或重新插拔"

echo
echo "=========== 3. 收尾 ==========="
kill $KIN_PID $LIDAR_PID $CORE_PID 2>/dev/null
sleep 2
kill -9 $KIN_PID $LIDAR_PID $CORE_PID 2>/dev/null
kill $RC_PID 2>/dev/null
sleep 1
kill -9 $RC_PID 2>/dev/null
echo "自检结束。上面每条都显示 average rate 即为正常。"
