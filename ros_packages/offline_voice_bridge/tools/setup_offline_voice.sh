#!/usr/bin/env bash
# 安装离线语音链路所需的组件：
#   1) Piper TTS 预编译二进制（语音合成）
#   2) 中文语音模型 zh_CN-huayan-medium
#   3) 检查 FunASR SenseVoiceSmall（可选，仅 /voice/asr_text 节点需要）
#
# 用法：bash setup_offline_voice.sh
set -u

PIPER_DIR="${PIPER_DIR:-$HOME/piper}"
VOICE_DIR="$PIPER_DIR/voices"
VOICE_NAME="${VOICE_NAME:-zh_CN-huayan-medium}"
SENSEVOICE_DIR="${SENSEVOICE_DIR:-$HOME/models/SenseVoiceSmall}"

PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
VOICE_BASE="https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium"

mkdir -p "$PIPER_DIR" "$VOICE_DIR"

echo "=== 1/3 下载 Piper TTS ==="
if [ -x "$PIPER_DIR/piper/piper" ]; then
  echo "已存在，跳过：$PIPER_DIR/piper/piper"
else
  wget -c -q --tries=10 --timeout=30 -O "$PIPER_DIR/piper_linux_x86_64.tar.gz" "$PIPER_URL" || {
    echo "下载 Piper 失败"; exit 1; }
  tar -xzf "$PIPER_DIR/piper_linux_x86_64.tar.gz" -C "$PIPER_DIR"
  echo "解压完成：$PIPER_DIR/piper/piper"
fi

echo "=== 2/3 下载中文语音模型 ==="
for f in "$VOICE_NAME.onnx" "$VOICE_NAME.onnx.json"; do
  if [ -s "$VOICE_DIR/$f" ]; then
    echo "已存在，跳过：$f"
  else
    wget -c -q --tries=10 --timeout=30 -O "$VOICE_DIR/$f" "$VOICE_BASE/$f" || {
      echo "下载 $f 失败"; exit 1; }
    echo "已下载：$f"
  fi
done

echo "=== 3/3 检查语音识别组件（可选）==="
if [ -d "$SENSEVOICE_DIR" ]; then
  echo "SenseVoiceSmall 已就绪：$SENSEVOICE_DIR"
else
  echo "未找到 $SENSEVOICE_DIR"
  echo "（仅当需要 /voice/asr_text 话题时才需要；任务默认使用 direct_asr 模式，可忽略）"
fi
python3 -c "import funasr" 2>/dev/null && echo "funasr 已安装" || echo "funasr 未安装：pip3 install --user funasr"

echo
echo "=== 验证语音合成 ==="
echo "你好，我是居家生活机器人。" | "$PIPER_DIR/piper/piper" \
  --model "$VOICE_DIR/$VOICE_NAME.onnx" --output_file /tmp/piper_check.wav >/dev/null 2>&1 \
  && ls -l /tmp/piper_check.wav \
  && echo "播放测试（应能听到声音）：aplay /tmp/piper_check.wav" \
  || echo "合成失败，请检查上面的输出"

echo
echo "完成。启动命令："
echo "  roslaunch offline_voice_bridge offline_voice_zh.launch"
