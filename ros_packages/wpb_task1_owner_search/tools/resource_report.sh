#!/usr/bin/env bash
# 资源体检：跑任务时**另开一个终端**执行，一次看清 CPU / 内存 / 显存 / 各节点占用。
#
#   bash resource_report.sh          # 单次快照
#   bash resource_report.sh 60       # 每 10 秒打一次，持续 60 秒（看峰值）
#
# 为什么要看这个：2026-10-09 排查发现 ⑦b+⑫ 这条链路上**所有模型都跑在 CPU**
# （姿态/ReID/人脸/ASR/TTS/相机搬运），GTX 1650 基本闲着。相机掉流又总出现在
# "模型加载/CPU 尖峰"附近，所以先量出到底谁在吃 CPU、有没有把 16 个线程吃满。

set -uo pipefail

ROUNDS=${1:-0}
INTERVAL=10

show_once() {
  echo "==================== $(date '+%F %T') ===================="
  echo "--- 负载 / 内存（MB）"
  uptime
  free -m | head -3

  echo "--- 显存（GTX 1650）"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu \
               --format=csv,noheader 2>&1 | head -2
    echo "    占用显存的进程："
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | head -10 | sed 's/^/      /'
  else
    echo "    （没有 nvidia-smi）"
  fi

  echo "--- CPU 占用前 12（%CPU / %MEM / 常驻内存）"
  ps -eo pcpu,pmem,rss,comm --sort=-pcpu 2>/dev/null | head -13

  echo "--- ROS 相关进程逐个体检（nodelet=相机、task_all_owners=任务、qwen=动作节点）"
  echo "    注：cpu% 是「从启动到现在的平均值」，多核会 >100%；看趋势请连续对比几轮"
  ps -eo pid,pcpu,pmem,rss,etime,args --sort=-pcpu 2>/dev/null \
    | grep -E "nodelet|rosmaster|roslaunch|move_base|amcl|lidar_loc|task_all_owners|qwen_action|kinect2|offline_asr|offline_tts|rviz|wp_manager|wpb_home_core" \
    | grep -v grep \
    | awk '{mb=$4/1024; printf "  pid=%-7s cpu%6s%%  mem%5s%%  %7.0fMB  up %-9s %s\n", $1, $2, $3, mb, $5, substr($0, index($0,$6))}' \
    | head -25
  echo "--- 只挑 torch/推理进程（看清是谁在算模型）"
  ps -eo pid,pcpu,rss,etime,comm,args 2>/dev/null \
    | grep -E "pt_main_thread|ultralytics|torch|onnxruntime|insight|sensevoice|piper|funasr" \
    | grep -v grep | head -12
  echo
}

show_once
if [ "$ROUNDS" -gt 0 ]; then
  for _ in $(seq 1 "$ROUNDS"); do
    sleep "$INTERVAL"
    show_once
  done
fi

cat <<'TXT'
==================== 当前各模型跑在哪（代码默认，实际以任务传参为准） ====================
  姿态/动作   YOLO11n-pose  /home/dyp/models/yolo11n-pose.pt (6.3MB)   CPU
              ↑ 任务用 _action_pose_device 传参，默认 cpu（日志会打 "device=cpu"）
  ReID        OSNet x0_25   models/reid/osnet_x0_25_msmt17.pth (9.3MB) CPU
              ↑ ~reid_device 默认 cpu；要上 GPU：_reid_device:=cuda:0
  人脸        InsightFace   ~/.insightface (30MB)                      CPU
              ↑ ~face_ctx_id 默认 -1（=CPUExecutionProvider）；上 GPU：_face_ctx_id:=0
  语音识别    SenseVoiceSmall /home/dyp/models/SenseVoiceSmall (901MB)  CPU
              ↑ offline_asr_node 的 ~device 默认 cpu
  语音合成    Piper zh_CN-huayan (63MB, onnx)                          CPU
  相机        kinect2_bridge（彩色/深度搬运）+ 点云 nodelet              CPU
  YOLO-World  yolov8s-world-person                                    GPU(cuda:0)
              ↑ 只有 ⑦（start_yolo 默认 true）会起；⑦b 显式 start_yolo:=false 不起
  结论：⑦b + ⑫ 这条链路上 GPU 基本闲着，16 个 CPU 线程承担全部推理与搬运。
TXT
