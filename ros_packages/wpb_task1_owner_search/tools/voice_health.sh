#!/usr/bin/env bash
# 语音链路体检：TTS（说话）与 ASR（听话）在不在、话题通不通。
#
# 为什么需要它：**bringup 里 start_asr 默认是 false**，只有 VS Code 的 ⑦/⑦b 显式带
# start_asr:=true。用终端起栈很容易忘记，症状是"机器人会说话，但注册时听不到你说话"，
# 任务只会播报一句"语音识别没有连接，请检查离线语音节点"。
#
#   bash voice_health.sh            # 体检
#   bash voice_health.sh --fix-cmd  # 只打印"怎么把 ASR 挂上来"的命令

set -uo pipefail
source /opt/ros/noetic/setup.bash 2>/dev/null || true
source "$HOME/catkin_ws/devel/setup.bash" 2>/dev/null || true

FIX_ASR='roslaunch offline_voice_bridge offline_voice_zh.launch \
  start_tts:=false start_sound_play:=false start_asr:=true \
  asr_window_seconds:=2.5'
FIX_TTS='roslaunch offline_voice_bridge offline_voice_zh.launch \
  start_asr:=false start_tts:=true start_sound_play:=false'
FIX_BOTH='roslaunch offline_voice_bridge offline_voice_zh.launch \
  start_tts:=true start_asr:=true start_sound_play:=true \
  asr_window_seconds:=2.5'

if [[ "${1:-}" == "--fix-cmd" ]]; then
  echo "# 缺 TTS（不会说话）："; echo "$FIX_TTS"
  echo "# 缺 ASR（听不见）："; echo "$FIX_ASR"
  echo "# 两个都缺："; echo "$FIX_BOTH"
  exit 0
fi

echo "==================== 语音链路体检 $(date '+%F %T') ===================="
nodes="$(rosnode list 2>&1)"
if [[ "$nodes" == *"ERROR"* || -z "$nodes" ]]; then
  echo "❌ rosnode list 失败：$nodes"
  echo "   → 栈没起来。先 robot.sh stop，再用 ⑦b 起栈（它带 start_asr:=true）"
  exit 1
fi

tts_ok=0; asr_ok=0
grep -q "/offline_tts_node" <<<"$nodes" && tts_ok=1
grep -q "/offline_asr_node" <<<"$nodes" && asr_ok=1

echo "--- 节点"
if (( tts_ok )); then echo "  ✅ /offline_tts_node（说话）"; else echo "  ❌ /offline_tts_node 不在（机器人不会说话）"; fi
if (( asr_ok )); then echo "  ✅ /offline_asr_node（听话）"; else echo "  ❌ /offline_asr_node 不在（注册会卡在'请说姓名'）"; fi

echo "--- 话题 /voice/say（任务说 → TTS 播）"
rostopic info /voice/say 2>&1 | sed -n '/Publishers/,$p' | head -8 | sed 's/^/  /'
echo "--- 话题 /voice/asr_text（ASR 识别 → 任务听）"
rostopic info /voice/asr_text 2>&1 | sed -n '/Publishers/,$p' | head -8 | sed 's/^/  /'

echo "--- 最近 ASR 识别窗口（有没有在把你说的话转成文字）"
tail -3 "$HOME/.ros/log/offline_asr_node.log" 2>/dev/null | sed 's/^/  /' || echo "  （没有 ASR 日志）"

echo
if (( tts_ok && asr_ok )); then
  echo "✅ 两个都在。若注册还是听不到：看上面 ASR 窗口日志有没有把你的话转成文字；"
  echo "   没有的话试  bash tools/repair_xbox_nui_audio.sh  （PulseAudio 麦克风路由）"
else
  echo "❌ 缺节点。原因提醒：bringup 里 start_asr 默认 false、start_tts 默认 true，"
  echo "   用终端起栈时若显式写了 start_tts:=false（或只带 start_asr:=true），就会出现现在这种情况。"
  echo "   修复（不用重启整个栈）："
  if (( ! tts_ok )); then echo "   —— 补 TTS（说话）："; echo "$FIX_TTS" | sed 's/^/   /'; fi
  if (( ! asr_ok )); then echo "   —— 补 ASR（听话）："; echo "$FIX_ASR" | sed 's/^/   /'; fi
  if (( ! tts_ok && ! asr_ok )); then echo "   —— 或两个都缺，直接："; echo "$FIX_BOTH" | sed 's/^/   /'; fi
  echo "   （想省事就直接用 VS Code ⑦b 重新起栈，它也带 start_asr:=true）"
fi
