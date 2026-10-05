#!/usr/bin/env bash
# 启智机器人统一入口脚本 —— 不用再开多个终端手动敲命令
#
# 用法：
#   bash ~/catkin_ws/competition/robot.sh stack     只启动机器人栈（前台，Ctrl+C 停止）
#   bash ~/catkin_ws/competition/robot.sh enroll    栈放后台 + 启动主人注册（前台）
#   bash ~/catkin_ws/competition/robot.sh task      栈放后台 + 启动任务节点（前台）
#   bash ~/catkin_ws/competition/robot.sh once      一条 roslaunch 启动全部（含任务节点）
#   bash ~/catkin_ws/competition/robot.sh stop      停掉所有本脚本启动的进程
#   bash ~/catkin_ws/competition/robot.sh status    查看当前状态（节点/话题/Ollama）
#
# 可选环境变量：
#   OWNER_COUNT=3     注册人数（enroll 模式，默认 1）
#   NAVIGATE=1        是否启用导航（task 模式，默认 0=不导航）
#   ENROLL=1          task 模式下启用【现场注册主人】而不是读照片
#   SHOW_RVIZ=1       是否开 RViz（默认 0）
#   NO_OLLAMA=1       跳过启动 Ollama

set -u

PKG=wpb_task1_owner_search
LOG_DIR=/tmp/robot_run
STACK_LOG="$LOG_DIR/stack.log"
PID_FILE="$LOG_DIR/stack.pid"

OWNER_COUNT="${OWNER_COUNT:-1}"
NAVIGATE="${NAVIGATE:-0}"
SHOW_RVIZ="${SHOW_RVIZ:-0}"

mkdir -p "$LOG_DIR"

say()  { printf '\033[1;36m>>>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m###\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ 环境
worker_env() {
    # shellcheck disable=SC1091
    source /opt/ros/noetic/setup.bash
    # shellcheck disable=SC1091
    source "$HOME/catkin_ws/devel/setup.bash"
    export PATH="$HOME/.local/bin:$PATH"
}

# 精确清理 ROS 进程：按“可执行文件名 / 脚本名”匹配 /proc/<pid>/cmdline，
# 而不是用 pkill -f 的宽泛模式（那种写法容易误伤无关进程）。
# 参数：INT（优雅停止）或 KILL（强制）
kill_ros_processes() {
    python3 - "${1:-INT}" <<'PY'
import os, signal, sys

sig = signal.SIGKILL if str(sys.argv[1]).upper() == "KILL" else signal.SIGINT
targets = {
    "roslaunch", "rosmaster", "rosout", "rosnode", "rostopic",
    "wpb_home_core", "wpb_home_lidar_filter", "kinect2_bridge",
    # nodelet 必须单独列上：kinect2_bridge / depth_image_proc 都是通过
    # nodelet manager 加载的，进程名其实是 "nodelet"。
    # 漏掉它会导致残留的 nodelet manager 一直占着 Kinect 的 USB 接口，
    # 下次启动时报 "did not claim interface 0" 然后初始化失败。
    "nodelet",
    "rplidarNode", "map_server", "move_base", "amcl", "wp_manager",
    "robot_state_publisher", "offline_tts_node.py", "offline_asr_node.py",
    "yoloworld_async.py", "yoloworld_debug_viewer_stable.py",
    "task1_find_owner_real.py", "owner_voice_reid_test.py",
    "person_reid_owner_test.py", "rviz",
}
me, parent = os.getpid(), os.getppid()
hit = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    ipid = int(pid)
    if ipid in (me, parent, 1):
        continue
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as fh:
            argv = [p.decode("utf-8", "replace") for p in fh.read().split(b"\0") if p]
    except OSError:
        continue
    if not argv:
        continue
    if {os.path.basename(a) for a in argv} & targets:
        try:
            os.kill(ipid, sig)
            hit.append(os.path.basename(argv[0]) if argv[0] else str(ipid))
        except OSError:
            pass
if hit:
    print("  已停止: " + ", ".join(sorted(set(hit))))
PY
}

# 前台运行一条命令，并保证“脚本被终止”时 roslaunch 及其子节点不残留。
# 注意：kill 掉 roslaunch 本身并不会杀掉它启动的节点，
# 所以最后再加一次按进程名的精确清理兜底。
run_foreground() {
    "$@" &
    local child=$!
    _stop_child() {
        kill -INT "$child" 2>/dev/null
        local i
        for i in $(seq 1 16); do
            kill -0 "$child" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "$child" 2>/dev/null
        kill_ros_processes INT
    }
    trap '_stop_child; trap - INT TERM EXIT; exit 130' INT
    trap '_stop_child; trap - INT TERM EXIT; exit 143' TERM
    trap '_stop_child' EXIT
    wait "$child"
    local rc=$?
    trap - INT TERM EXIT
    return "$rc"
}

stack_running() {
    [ -f "$PID_FILE" ] || return 1
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

roscore_running() {
    pgrep -x rosmaster >/dev/null 2>&1
}

# ------------------------------------------------------------------ Ollama
start_ollama() {
    if [ "${NO_OLLAMA:-0}" = "1" ]; then
        say "按 NO_OLLAMA=1 跳过 Ollama"
        return 0
    fi
    if curl -sS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        say "Ollama 已在运行"
        return 0
    fi
    say "启动 Ollama（动作识别用的视觉模型）"
    setsid nohup env PATH="$HOME/.local/bin:/usr/bin:/bin" \
        OLLAMA_HOST=127.0.0.1:11434 \
        "$HOME/.local/bin/ollama" serve > "$LOG_DIR/ollama.log" 2>&1 &
    local i
    for i in $(seq 1 20); do
        sleep 1
        if curl -sS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
            say "Ollama 就绪"
            return 0
        fi
    done
    warn "Ollama 启动超时，看 $LOG_DIR/ollama.log（动作识别会降级，其余不受影响）"
    return 1
}

# ------------------------------------------------------------------ 机器人栈
rviz_arg() {
    if [ "$SHOW_RVIZ" = "1" ]; then echo "start_rviz:=true"; else echo "start_rviz:=false"; fi
}

start_stack_background() {
    local extra_args="${1:-}"
    if stack_running; then
        say "机器人栈已在运行（pid $(cat "$PID_FILE")）"
        return 0
    fi
    if roscore_running; then
        warn "检测到残留的 rosmaster，请先执行：bash $0 stop"
        die "为避免端口冲突已退出"
    fi

    say "启动机器人栈（底盘/雷达/Kinect/地图/定位/导航/行人检测/语音）"
    local rviz
    rviz="$(rviz_arg)"

    setsid nohup bash -c "
        source /opt/ros/noetic/setup.bash
        source \$HOME/catkin_ws/devel/setup.bash
        exec roslaunch $PKG task1_owner_search_bringup.launch $rviz $extra_args
    " > "$STACK_LOG" 2>&1 &

    echo $! > "$PID_FILE"
    say "栈的日志：$STACK_LOG"
}

wait_for_stack() {
    local wait_seconds="${1:-180}"
    local elapsed=0
    say "等待传感器就绪（最多 ${wait_seconds}s，YOLO 首次加载较慢）..."

    while [ "$elapsed" -lt "$wait_seconds" ]; do
        local ok=0
        if timeout 5 rostopic hz /kinect2/qhd/image_color_rect 2>/dev/null | grep -q "average rate"; then
            ok=$((ok + 1))
        fi
        if timeout 5 rostopic hz /perception/person_detections_2d 2>/dev/null | grep -q "average rate"; then
            ok=$((ok + 1))
        fi
        if [ "$ok" -ge 2 ]; then
            say "相机与行人检测都已出数据，栈就绪"
            return 0
        fi
        printf '.'
        sleep 6
        elapsed=$((elapsed + 6))
    done
    echo
    warn "等待超时，可能原因："
    warn "  1) YOLO 还在加载 CLIP 权重，再等 1 分钟"
    warn "  2) 设备没插好，检查 /dev/ftdi /dev/rplidar 与 Kinect 的 USB"
    warn "  3) 看日志：$STACK_LOG"
    return 1
}

stop_all() {
    say "停止机器人相关进程"
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null
        sleep 3
        kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
        rm -f "$PID_FILE"
    fi
    kill_ros_processes INT
    sleep 3
    kill_ros_processes KILL
    pkill -9 -x rosmaster 2>/dev/null
    pkill -9 -x rosout 2>/dev/null
    say "已停止（Ollama 保留运行；要停它请用：pkill -x ollama）"
}

show_status() {
    worker_env
    if pgrep -x rosmaster >/dev/null 2>&1; then
        say "ROS master : 运行中"
    else
        say "ROS master : 未运行"
    fi
    say "Ollama     : $(curl -sS --max-time 3 http://127.0.0.1:11434/api/version 2>/dev/null || echo 未运行)"
    if stack_running; then
        say "机器人栈   : 运行中（pid $(cat "$PID_FILE")）"
    else
        say "机器人栈   : 未运行"
    fi
    echo
    if pgrep -x rosmaster >/dev/null 2>&1; then
        echo "--- 节点 ---"
        timeout 8 rosnode list 2>/dev/null | sort | sed 's/^/  /'
        echo "--- 关键话题频率 ---"
        local t
        for t in /odom /scan /kinect2/qhd/image_color_rect /perception/person_detections_2d; do
            printf '  %-38s ' "$t"
            timeout 5 rostopic hz "$t" 2>/dev/null | grep -m1 "average rate" || echo "无数据"
        done
    fi
}

# ------------------------------------------------------------------ 主流程
worker_env

case "${1:-}" in
  stack)
    start_ollama || true
    [ "$SHOW_RVIZ" = "1" ] || say "RViz 已关闭（要开请用 SHOW_RVIZ=1）"
    say "前台运行机器人栈，按 Ctrl+C 停止"
    run_foreground roslaunch "$PKG" task1_owner_search_bringup.launch "$(rviz_arg)"
    ;;

  enroll)
    start_ollama || true
    # 注册工具自己会启动一整套语音（TTS+ASR），所以栈里不要再起语音，
    # 否则会出现两个同名节点互相顶掉。
    start_stack_background "start_voice:=false"
    wait_for_stack 180 || true
    say "启动主人注册（注册 $OWNER_COUNT 位；不导航）"
    say "注册完成后进入识别循环，站到镜头前应听到自己的名字"
    run_foreground roslaunch "$PKG" owner_voice_reid_test.launch \
        owner_count:="$OWNER_COUNT" navigate_enabled:=false start_navigation:=false \
        "$(rviz_arg)"
    ;;

  task)
    start_ollama || true
    # 现场注册需要 ASR 在线。关键点：start_asr 只有在 start_voice=true 时才有效，
    # 而 bringup.launch 里 start_voice 默认就是 true、start_asr 默认 false，
    # 所以这里必须显式把 start_asr 打开，让【栈】去启动 ASR；
    # 任务节点那边保持 start_voice=false，避免重复启动语音节点。
    if [ "${ENROLL:-0}" = "1" ]; then
        start_stack_background "start_asr:=true"
    else
        start_stack_background
    fi
    wait_for_stack 180 || true
    task_args=()
    [ "$NAVIGATE" = "1" ] && task_args+=(navigate_enabled:=true) || task_args+=(navigate_enabled:=false)
    if [ "${ENROLL:-0}" = "1" ]; then
        # 现场注册：不读 data/owner 照片，启动时语音问名字并采集人脸
        task_args+=(owner_enrollment_enabled:=true)
        say "启动任务节点【现场注册模式】——不会读取 data/owner 的照片"
        say "画面里站一个人，听到“请在我前方说出您的名字”后报名字"
        # 开跑前确认 ASR 真的在线，避免任务节点因为等不到发布者而直接退出
        if ! timeout 10 rostopic info /voice/asr_text 2>/dev/null | sed -n '/Publishers:/,$p' | grep -q '\*'; then
            warn "警告：/voice/asr_text 还没有发布者，现场注册会立刻失败"
            warn "  最常见原因：栈是用【不带 start_asr】的方式启动的（比如先跑过 stack 模式）"
            warn "  处理：先执行 bash $0 stop，再重新跑本命令"
            warn "  如果是别的机器/进程占着麦克风，也会出现同样现象"
        fi
    else
        say "启动任务节点（身份来自 data/owner 里的照片）"
    fi
    [ "$NAVIGATE" = "1" ] || say "导航已关闭，机器人会原地扫描找人"
    run_foreground roslaunch "$PKG" task1_owner_search_task_only.launch "${task_args[@]}"
    echo
    say "任务节点已退出。注意：机器人栈仍在后台运行（方便你重新跑任务）"
    say "  想重新跑任务：再执行一次同样的命令即可"
    say "  想全部停掉  ：bash $0 stop"
    ;;

  once)
    start_ollama || true
    say "一条命令启动全部（含任务节点）；全栈重启较慢，调试建议用 task 模式"
    run_foreground roslaunch "$PKG" task1_owner_search_real.launch
    ;;

  stop)   stop_all ;;
  status) show_status ;;

  *)
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
    ;;
esac
