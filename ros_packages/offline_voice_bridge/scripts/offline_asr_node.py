#!/usr/bin/env python3
# coding: utf-8
"""Offline Chinese ASR node: microphone -> FunASR SenseVoiceSmall -> /voice/asr_text.

循环抓取固定长度音频窗口，音量超过阈值时做识别，并把文本发布到
/voice/asr_text（std_msgs/String）。任务节点在
electrical_switch_instruction_source: ros_topic 模式下会订阅这个话题。
"""

import audioop
import math
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import rospy
from std_msgs.msg import Bool, String


class OfflineAsrNode(object):
    def __init__(self):
        self.asr_topic = rospy.get_param("~asr_topic", "/voice/asr_text")
        self.capture_device = str(
            rospy.get_param("~capture_device", rospy.get_param("/asr/capture_device", "default"))
        )
        # PulseAudio 采集参数，由 wpb_task1_owner_search 的 launch 传入。
        # 默认 backend=off，保持单独启动时的原有行为不变。
        self.capture_backend = str(rospy.get_param("~capture_backend", "off")).strip().lower()
        self.capture_source = str(rospy.get_param("~capture_source", "")).strip()
        self.capture_volume = str(rospy.get_param("~capture_volume", "100%")).strip() or "100%"
        try:
            self.input_gain = float(rospy.get_param("~input_gain", 1.0))
        except (TypeError, ValueError):
            self.input_gain = 1.0
        self.energy_threshold = int(
            rospy.get_param("~energy_threshold", rospy.get_param("/asr/energy_threshold", 300))
        )
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))
        self.window_seconds = max(0.5, float(rospy.get_param("~window_seconds", 4.0)))
        # ---- 重叠窗口 ----
        # 原来是"每 window_seconds 秒切一段、独立识别"。问题是切点跟说话时机
        # 无关，一句话正好压在分界线上就会被劈成两个碎片——实测：
        #   ASR window 759 (RMS=443): 姓名。
        #   ASR window 760 (RMS=630): 名章3。
        # 两个半句都认不出来，用户只能重说，直到整句碰巧落进同一段里。
        # 现在改成：每次只新录 capture_block_seconds，送去识别的是
        # "上一段 + 这一段"拼起来的完整窗口（相邻窗口重叠 50%）。
        # 于是只要一句话不超过 (窗口 - 步长)，就必定完整落在某一个窗口里。
        # 关掉这个开关（~window_overlap_enabled:=false）就退回原来的行为。
        self.overlap_enabled = bool(
            rospy.get_param("~window_overlap_enabled", True)
        )
        if self.overlap_enabled:
            # arecord 的 -d 只吃整数秒，所以步长取整到秒；
            # 窗口 = 2 × 步长，保证重叠 50%。
            self.capture_block_seconds = max(
                1, int(math.ceil(self.window_seconds / 2.0))
            )
            self.frame_seconds = self.capture_block_seconds * 2
        else:
            self.capture_block_seconds = max(1, int(round(self.window_seconds)))
            self.frame_seconds = self.capture_block_seconds
        # 同一句话会在相邻两个重叠窗口里各出现一次，去重窗口要比步长宽一点
        self.dedup_seconds = max(
            0.5, float(rospy.get_param("~dedup_seconds", 2.5))
        )
        # ---- 自适应门限 + 裁静音 + 归一化 ----
        # 固定增益在安静的实验室好用，但比赛现场底噪一高，增益会把噪声一起
        # 抬起来，"哪段算说话"的门限就废了（全被当成语音送去识别 -> 满屏幻觉）。
        # 所以改成跟本底走：
        #   门限 = max(固定下限, 本底噪声 × 倍数)
        # 再把句子前后的静音裁掉、整段归一化到固定强度，送进模型的信号强度
        # 就跟场地、麦克风无关了。
        self.adaptive_gate = bool(rospy.get_param("~adaptive_gate", True))
        self.noise_gate_ratio = max(
            1.0, float(rospy.get_param("~noise_gate_ratio", 2.5))
        )
        self.noise_history_len = max(5, int(rospy.get_param("~noise_history_len", 40)))
        self.noise_percentile = min(
            90.0, max(1.0, float(rospy.get_param("~noise_percentile", 20.0)))
        )
        # 门限上限：万一现场持续很吵，别让门限涨到把真人说话也挡掉
        self.max_gate = max(
            float(self.energy_threshold),
            float(rospy.get_param("~max_gate", 1500.0)),
        )
        self.crop_silence = bool(rospy.get_param("~crop_silence", True))
        self.crop_margin_seconds = max(
            0.0, float(rospy.get_param("~crop_margin_seconds", 0.25))
        )
        self.normalize_audio = bool(rospy.get_param("~normalize_audio", True))
        self.normalize_target_rms = max(
            100.0, float(rospy.get_param("~normalize_target_rms", 1200.0))
        )
        self.normalize_max_gain = max(
            1.0, float(rospy.get_param("~normalize_max_gain", 12.0))
        )
        self._rms_history = []
        # ---- 声学级静音：TTS 播报期间停止录音 ----
        # TTS 节点在 /voice/speaking 上发布 Bool：true=正在播报。
        # 播报期间不录音，录音窗口内检测到播报开始也会丢弃该窗口，
        # 这样机器人就不会把自己说的话当成用户回答（文本级回声过滤
        # 只能事后补救，声学级静音从源头掐掉）。
        self.speaking_topic = rospy.get_param("~speaking_topic", "/voice/speaking")
        self.squelch_during_tts = bool(rospy.get_param("~squelch_during_tts", True))
        # 安全阀：万一 TTS 节点崩了没落旗，超过这个时长就自动恢复录音，
        # 否则 ASR 会被"卡住"永远不录。
        self.max_speaking_seconds = max(
            5.0, float(rospy.get_param("~max_speaking_seconds", 60.0))
        )
        self.tts_speaking = False
        self._last_tts_true_time = 0.0
        self._speaking_started_at = 0.0
        self.model_path = os.path.expanduser(
            rospy.get_param(
                "~model_path",
                os.path.join(os.path.expanduser("~"), "models", "SenseVoiceSmall"),
            )
        )
        self.language = str(rospy.get_param("~language", "zh")).strip() or "auto"
        self.device = str(rospy.get_param("~device", "cpu")).strip() or "cpu"
        self.use_itn = bool(rospy.get_param("~use_itn", True))
        self.min_text_length = max(1, int(rospy.get_param("~min_text_length", 2)))
        self.publish_empty = bool(rospy.get_param("~publish_empty", False))
        self.keep_wav = bool(rospy.get_param("~keep_wav", False))
        self.tmp_dir = os.path.expanduser(rospy.get_param("~tmp_dir", "/dev/shm"))

        self._model = None
        self._model_lock = threading.Lock()
        self._counter = 0
        # 启动时先跑一次空推理：SenseVoice 首次 generate() 要 11~14 秒，
        # 不预热的话这段时间录音线程是整块卡住的——起栈后马上跑注册，
        # 用户前几遍说话会被整个吃掉（实测："语音识别很久没反应，
        # 起栈久一点再跑就正常"）。预热把这段开销挪到起栈阶段付掉。
        self.warmup_enabled = bool(rospy.get_param("~warmup", True))

        self.setup_pulse_capture()

        self.pub = rospy.Publisher(self.asr_topic, String, queue_size=10)
        self.speaking_sub = rospy.Subscriber(
            self.speaking_topic, Bool, self.speaking_callback, queue_size=10
        )
        self._worker = threading.Thread(target=self._worker_loop)
        self._worker.daemon = True
        self._worker.start()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "Offline ASR ready: topic=%s device=%s model=%s threshold=%d gain=%.2f "
            "tts_squelch=%s(%s)",
            self.asr_topic,
            self.capture_device,
            self.model_path,
            self.energy_threshold,
            self.input_gain,
            self.squelch_during_tts,
            self.speaking_topic,
        )

    def speaking_callback(self, message):
        """TTS 播报状态：true=机器人正在说话，这段时间不录音。"""
        active = bool(message.data)
        previous = self.tts_speaking
        if not active:
            self._speaking_started_at = 0.0
        self.tts_speaking = active
        if active:
            self._last_tts_true_time = time.time()
            if not self._speaking_started_at:
                self._speaking_started_at = self._last_tts_true_time
        # 只在状态翻转时打日志，方便排查"标志卡住不录音"的问题
        if active and not previous:
            rospy.loginfo("TTS 播报开始，暂停录音")
        elif not active and previous:
            rospy.loginfo("TTS 播报结束，恢复语音采集")

    # ---------- PulseAudio 采集源 ----------
    def _pactl(self, args):
        """执行一条 pactl 命令，返回 (成功, 输出)。失败只记录不抛异常。"""
        try:
            proc = subprocess.run(
                ["pactl"] + list(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5.0,
            )
        except Exception as exc:
            return False, str(exc)
        output = proc.stdout.decode("utf-8", errors="replace").strip()
        return proc.returncode == 0, output

    def _find_pulse_source(self):
        """在 PulseAudio 输入源里找 Kinect 麦克风阵列，找不到就退而求其次。"""
        ok, output = self._pactl(["list", "short", "sources"])
        if not ok:
            return ""
        candidates = []
        for line in output.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[1].strip()
            # monitor 源是扬声器回环，不能当麦克风用
            if not name or name.endswith(".monitor"):
                continue
            candidates.append(name)
        for keyword in ("Xbox_NUI_Sensor", "usb"):
            for name in candidates:
                if keyword.lower() in name.lower():
                    return name
        return candidates[0] if candidates else ""

    def setup_pulse_capture(self):
        """backend=pulse/auto 时设置默认采集源、取消静音并按参数调音量。

        任何一步失败都只打日志、不影响节点启动；此时仍按 capture_device 采集。
        """
        if self.capture_backend not in ("pulse", "pulseaudio", "auto"):
            return
        if shutil.which("pactl") is None:
            rospy.logwarn(
                "capture_backend=%s 但找不到 pactl，继续使用 capture_device=%s",
                self.capture_backend,
                self.capture_device,
            )
            return

        source = self.capture_source or self._find_pulse_source()
        if not source:
            rospy.logwarn(
                "没有找到可用的 PulseAudio 输入源，继续使用 capture_device=%s",
                self.capture_device,
            )
            return

        for args in (
            ["set-default-source", source],
            ["set-source-mute", source, "0"],
            ["set-source-volume", source, self.capture_volume],
        ):
            ok, output = self._pactl(args)
            if not ok:
                rospy.logwarn("pactl %s 失败: %s", " ".join(args), output)

        rospy.loginfo(
            "PulseAudio 采集源已设置: source=%s volume=%s gain=%.2f",
            source,
            self.capture_volume,
            self.input_gain,
        )

    # ---------- ASR model ----------
    def ensure_model(self):
        if self._model is not None:
            return True
        with self._model_lock:
            if self._model is not None:
                return True
            if os.path.sep in self.model_path and not os.path.exists(self.model_path):
                rospy.logerr("SenseVoice model path does not exist: %s", self.model_path)
                return False
            try:
                from funasr import AutoModel
            except ImportError as exc:
                rospy.logerr("funasr is not available: %s", exc)
                return False
            try:
                start = time.time()
                self._model = AutoModel(
                    model=self.model_path,
                    trust_remote_code=False,
                    device=self.device,
                    disable_update=True,
                )
                rospy.loginfo("SenseVoice model loaded in %.2fs", time.time() - start)
                return True
            except Exception as exc:
                self._model = None
                rospy.logerr("Failed to load SenseVoice model: %s", exc)
                return False

    @staticmethod
    def clean_text(raw_text):
        text = str(raw_text or "")
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            return rich_transcription_postprocess(text).strip()
        except Exception:
            import re

            return re.sub(r"<\|[^|]*\|>", "", text).strip()

    def transcribe(self, pcm_bytes):
        if not self.ensure_model():
            return ""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return ""
        results = self._model.generate(
            input=samples,
            fs=self.sample_rate,
            cache={},
            language=self.language,
            use_itn=self.use_itn,
            batch_size_s=60,
        )
        if not results:
            return ""
        return self.clean_text(results[0].get("text", ""))

    def warmup_model(self):
        """启动时用一小段静音先跑一遍，把首次推理的十几秒提前付掉。

        返回 True/False（失败不影响正常使用，只是第一句会慢）。
        """
        if not self.ensure_model():
            rospy.logwarn("ASR 预热跳过：模型没就绪")
            return False
        try:
            seconds = max(0.5, float(rospy.get_param("~warmup_seconds", 1.0)))
            samples = np.zeros(int(self.sample_rate * seconds), dtype=np.float32)
            start = time.time()
            # 纯静音，结果丢弃；只为触发一次真正的推理路径
            self._model.generate(
                input=samples,
                fs=self.sample_rate,
                cache={},
                language=self.language,
                use_itn=self.use_itn,
                batch_size_s=60,
            )
            rospy.loginfo(
                "ASR 预热完成：首次推理 %.2fs（这笔开销已经挪到起栈阶段，"
                "现场不会再卡第一句）",
                time.time() - start,
            )
            return True
        except Exception as exc:
            rospy.logwarn("ASR 预热失败（不影响使用，只是第一句会慢）：%s", exc)
            return False

    # ---------- audio capture loop ----------
    def apply_input_gain(self, pcm_bytes):
        """识别前对 PCM 做软件放大，并按 int16 范围裁剪。gain<=1 时原样返回。"""
        if self.input_gain <= 1.0 or not pcm_bytes:
            return pcm_bytes
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return pcm_bytes
        samples *= self.input_gain
        np.clip(samples, -32768.0, 32767.0, out=samples)
        return samples.astype(np.int16).tobytes()

    # ---------- 自适应门限 / 裁静音 / 归一化 ----------
    def remember_rms(self, rms):
        self._rms_history.append(float(rms))
        if len(self._rms_history) > self.noise_history_len:
            del self._rms_history[: len(self._rms_history) - self.noise_history_len]

    def noise_floor(self):
        """本底噪声估计：取最近若干段的低分位数（说话段是少数，不影响它）。"""
        if not self._rms_history:
            return 0.0
        recent = sorted(self._rms_history)
        index = int(len(recent) * self.noise_percentile / 100.0)
        index = max(0, min(len(recent) - 1, index))
        return float(recent[index])

    def current_gate(self):
        """当前该用多少能量门限。安静环境等于原来的固定值；吵了自动抬高。"""
        if not self.adaptive_gate or not self._rms_history:
            return float(self.energy_threshold)
        gate = self.noise_floor() * self.noise_gate_ratio
        return min(self.max_gate, max(float(self.energy_threshold), gate))

    def crop_to_active(self, pcm_bytes, level):
        """把句子前后的静音裁掉（两端各留一点余量），让模型只看有效语音。"""
        if not self.crop_silence or not pcm_bytes:
            return pcm_bytes
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples.size == 0:
            return pcm_bytes
        active = np.nonzero(np.abs(samples) > max(1.0, level))[0]
        if active.size == 0:
            return pcm_bytes
        margin = int(self.crop_margin_seconds * self.sample_rate)
        start = max(0, int(active[0]) - margin)
        end = min(samples.size, int(active[-1]) + margin)
        return samples[start:end].tobytes()

    def normalize_pcm(self, pcm_bytes):
        """把这段音频缩放到固定强度，抵消场地/麦克风带来的电平差异。"""
        if not self.normalize_audio or not pcm_bytes:
            return pcm_bytes
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return pcm_bytes
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms <= 1.0:
            return pcm_bytes
        gain = min(self.normalize_max_gain, self.normalize_target_rms / rms)
        if gain <= 1.02:
            return pcm_bytes
        samples *= gain
        np.clip(samples, -32768.0, 32767.0, out=samples)
        return samples.astype(np.int16).tobytes()

    def record_window(self, raw_path, seconds=None):
        duration = max(1, int(round(self.window_seconds if seconds is None else seconds)))
        cmd = [
            "arecord",
            "-q",
            "-D",
            self.capture_device,
            "-d",
            str(duration),
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-t",
            "raw",
        ]
        with open(raw_path, "wb") as raw_file:
            proc = subprocess.Popen(cmd, stdout=raw_file, stderr=subprocess.PIPE)
            _, stderr = proc.communicate(timeout=duration + 10.0)
        if proc.returncode != 0:
            raise RuntimeError(
                "arecord exited with %d: %s"
                % (proc.returncode, stderr.decode("utf-8", errors="replace").strip()[-200:])
            )

    def _worker_loop(self):
        raw_path = os.path.join(self.tmp_dir, "offline_asr_window.raw")
        failures = 0
        previous_pcm = b""
        last_text = ""
        last_text_time = 0.0
        # 先把模型加载 + 首次推理的开销付掉，再开始录音
        if self.warmup_enabled:
            self.warmup_model()
        rospy.loginfo(
            "ASR 录音模式：%s（每 %.0fs 录一段，识别 %.0fs 音频）",
            "重叠窗口" if self.overlap_enabled else "整段窗口",
            self.capture_block_seconds,
            self.frame_seconds,
        )
        while not rospy.is_shutdown():
            # 声学级静音：机器人正在说话时完全不录音，
            # 既不采到自己播报的声音，也不浪费识别算力。
            if self.squelch_during_tts and self.tts_speaking:
                # 安全阀：TTS 节点异常退出没落旗时，超时自动恢复录音
                if (
                    self._speaking_started_at
                    and time.time() - self._speaking_started_at
                    > self.max_speaking_seconds
                ):
                    rospy.logerr(
                        "播报标志已持续 %.0fs 未复位，判定 TTS 异常，强制恢复录音",
                        time.time() - self._speaking_started_at,
                    )
                    self.tts_speaking = False
                    self._speaking_started_at = 0.0
                # 播报期间不保留上一段：免得把机器人的声音拼进下一个窗口
                previous_pcm = b""
                rospy.sleep(0.05)
                continue

            window_started = time.time()
            try:
                self.record_window(raw_path, self.capture_block_seconds)
                failures = 0
            except Exception as exc:
                failures += 1
                rospy.logerr("Microphone capture failed (%d): %s", failures, exc)
                rospy.sleep(min(10.0, 1.0 * failures))
                continue
            window_finished = time.time()

            # 录音过程中机器人开始播报 -> 这段已经被污染，整段丢弃。
            # （只看当前标志不够：播报可能在本窗口中途开始又结束。）
            if self.squelch_during_tts and (
                self.tts_speaking or self._last_tts_true_time >= window_started
            ):
                rospy.loginfo_throttle(
                    2.0,
                    "丢弃一段录音窗口：期间机器人开始播报（窗口 %.1fs）",
                    window_finished - window_started,
                )
                previous_pcm = b""
                if not self.keep_wav:
                    try:
                        os.unlink(raw_path)
                    except OSError:
                        pass
                continue

            try:
                with open(raw_path, "rb") as raw_file:
                    raw = raw_file.read()
            except IOError:
                continue
            finally:
                if not self.keep_wav:
                    try:
                        os.unlink(raw_path)
                    except OSError:
                        pass

            if not raw:
                continue
            raw = self.apply_input_gain(raw)
            # 重叠模式：识别的是"上一段 + 这一段"，相当于窗口每步只前进一段，
            # 相邻窗口重叠 50%。这样一句话不会因为正好跨在切点上而被劈开。
            if self.overlap_enabled and previous_pcm:
                frame_pcm = previous_pcm + raw
            else:
                frame_pcm = raw
            if self.overlap_enabled:
                previous_pcm = raw
            rms = audioop.rms(frame_pcm, 2)
            gate = self.current_gate()
            self.remember_rms(rms)
            self._counter += 1
            if rms < gate and not self.publish_empty:
                continue

            # 门限之上的才当"有人在说话"：先裁掉前后静音，再归一化到固定强度，
            # 这样送进模型的信号强度跟场地/麦克风无关
            # 裁剪门限取"门限的 6 成"：明显低于语音、又略高于噪声本底，
            # 不至于把一个字的头尾切掉
            speech_pcm = self.crop_to_active(frame_pcm, max(100.0, gate * 0.6))
            speech_pcm = self.normalize_pcm(speech_pcm)

            try:
                start = time.time()
                text = self.transcribe(speech_pcm)
            except Exception as exc:
                rospy.logerr("ASR transcription failed: %s", exc)
                continue

            if len(text) < self.min_text_length:
                if self.publish_empty and text:
                    self.pub.publish(String(data=text))
                continue

            # 同一句话会在相邻两个重叠窗口里各出现一次，短时间去重一次。
            # 用去掉标点后的文本比较，免得"跳过"和"跳过。"被当成两句。
            now = time.time()
            compact = self.clean_text(text)
            if compact and compact == last_text and (now - last_text_time) <= self.dedup_seconds:
                rospy.loginfo_throttle(
                    2.0, "ASR 重复（重叠窗口）：%s，忽略", text
                )
                last_text_time = now
                continue
            last_text = compact
            last_text_time = now

            rospy.loginfo(
                "ASR window %d (RMS=%d, gate=%.0f, 本底=%.0f, %.1fs): %s (%.2fs)",
                self._counter,
                rms,
                gate,
                self.noise_floor(),
                len(speech_pcm) / max(1, self.sample_rate * self.channels * 2),
                text,
                time.time() - start,
            )
            self.pub.publish(String(data=text))

    def shutdown(self):
        pass


def main():
    rospy.init_node("offline_asr_node")
    OfflineAsrNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
