# offline_voice_bridge（自建替代版）

这是厂商 `offline_voice_bridge` 包的替代实现，用于
`wpb_task1_owner_search` 的离线中文语音链路。**包名、launch 文件名、
参数名与话题名都与原包保持一致**，所以任务包的 launch 文件不需要任何改动。

## 提供什么

| 功能 | 实现 | 说明 |
| --- | --- | --- |
| 语音合成 | `scripts/offline_tts_node.py` | 订阅 `/voice/say`，用 Piper 合成后用 aplay 播放 |
| 语音识别（可选） | `scripts/offline_asr_node.py` | 麦克风 → FunASR SenseVoiceSmall → `/voice/asr_text` |
| 启动入口 | `launch/offline_voice_zh.launch` | 参数名与原包一致 |
| 安装脚本 | `tools/setup_offline_voice.sh` | 下载 Piper 与中文语音模型 |

语音合成引擎与原包一致（都是 PiperTTS），所以音色风格对齐。
语音识别换成了 SenseVoiceSmall，不再依赖 faster-whisper。

## 安装

```bash
bash ~/catkin_ws/src/offline_voice_bridge/tools/setup_offline_voice.sh
```

会下载：

- Piper 预编译二进制 → `~/piper/piper/piper`（约 25 MB）
- 中文语音模型 `zh_CN-huayan-medium` → `~/piper/voices/`（约 63 MB）

## 使用

```bash
# 只跑语音合成
roslaunch offline_voice_bridge offline_voice_zh.launch

# 同时启用识别节点（发布 /voice/asr_text）
roslaunch offline_voice_bridge offline_voice_zh.launch start_asr:=true

# 手动测试
rostopic pub -1 /voice/say std_msgs/String "data: '我已经识别到主人。'"
```

## 常见问题

**没有声音**：`aplay -l` 查看实际声卡。若系统默认输出是 HDMI（显示器），
需要指定板载声卡，例如：

```bash
roslaunch offline_voice_bridge offline_voice_zh.launch \
  tts_speaker_device:=plughw:CARD=Generic_1,DEV=0
```

**识别节点报设备错误**：`asr_capture_device` 要填 `arecord -l` 里列出的采集设备，
本机可用的是 `plughw:CARD=Generic_1,DEV=0`。
