#!/usr/bin/env python3
# coding: utf-8
"""Offline Chinese ASR node: microphone -> FunASR SenseVoiceSmall -> /voice/asr_text.

循环抓取固定长度音频窗口，音量超过阈值时做识别，并把文本发布到
/voice/asr_text（std_msgs/String）。任务节点在
electrical_switch_instruction_source: ros_topic 模式下会订阅这个话题。
"""

import audioop
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import rospy
from std_msgs.msg import String


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

        self.setup_pulse_capture()

        self.pub = rospy.Publisher(self.asr_topic, String, queue_size=10)
        self._worker = threading.Thread(target=self._worker_loop)
        self._worker.daemon = True
        self._worker.start()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "Offline ASR ready: topic=%s device=%s model=%s threshold=%d gain=%.2f",
            self.asr_topic,
            self.capture_device,
            self.model_path,
            self.energy_threshold,
            self.input_gain,
        )

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

    def record_window(self, raw_path):
        cmd = [
            "arecord",
            "-q",
            "-D",
            self.capture_device,
            "-d",
            str(max(1, int(round(self.window_seconds)))),
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
            _, stderr = proc.communicate(timeout=self.window_seconds + 10.0)
        if proc.returncode != 0:
            raise RuntimeError(
                "arecord exited with %d: %s"
                % (proc.returncode, stderr.decode("utf-8", errors="replace").strip()[-200:])
            )

    def _worker_loop(self):
        raw_path = os.path.join(self.tmp_dir, "offline_asr_window.raw")
        failures = 0
        while not rospy.is_shutdown():
            try:
                self.record_window(raw_path)
                failures = 0
            except Exception as exc:
                failures += 1
                rospy.logerr("Microphone capture failed (%d): %s", failures, exc)
                rospy.sleep(min(10.0, 1.0 * failures))
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
            rms = audioop.rms(raw, 2)
            self._counter += 1
            if rms < self.energy_threshold and not self.publish_empty:
                continue

            try:
                start = time.time()
                text = self.transcribe(raw)
            except Exception as exc:
                rospy.logerr("ASR transcription failed: %s", exc)
                continue

            if len(text) < self.min_text_length:
                if self.publish_empty and text:
                    self.pub.publish(String(data=text))
                continue

            rospy.loginfo("ASR window %d (RMS=%d): %s (%.2fs)", self._counter, rms, text, time.time() - start)
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
