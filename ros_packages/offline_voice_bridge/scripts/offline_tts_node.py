#!/usr/bin/env python3
# coding: utf-8
"""Offline Chinese TTS node: /voice/say -> Piper -> speaker.

订阅 std_msgs/String，用 Piper 把中文文本合成 wav，再用 aplay 播放。
播放串行执行，避免多句提示音叠在一起。
"""

import collections
import os
import subprocess
import threading
import time

import rospy
from std_msgs.msg import Bool, String


class OfflineTtsNode(object):
    def __init__(self):
        self.say_topic = rospy.get_param("~say_topic", "/voice/say")
        self.piper_binary = os.path.expanduser(
            rospy.get_param("~piper_binary", os.path.join(os.path.expanduser("~"), "piper", "piper", "piper"))
        )
        self.voice_model = os.path.expanduser(
            rospy.get_param(
                "~voice_model",
                os.path.join(os.path.expanduser("~"), "piper", "voices", "zh_CN-huayan-medium.onnx"),
            )
        )
        self.voice_config = str(rospy.get_param("~voice_config", "")).strip()
        self.player = str(rospy.get_param("~player", "aplay")).strip() or "aplay"
        self.speaker_device = str(rospy.get_param("~speaker_device", "")).strip()
        self.tmp_dir = os.path.expanduser(rospy.get_param("~tmp_dir", "/dev/shm"))
        self.keep_wav = bool(rospy.get_param("~keep_wav", False))
        self.length_scale = float(rospy.get_param("~length_scale", 1.0))
        self.synthesis_timeout = float(rospy.get_param("~synthesis_timeout", 30.0))
        self.playback_timeout = float(rospy.get_param("~playback_timeout", 60.0))
        self.max_pending = max(1, int(rospy.get_param("~max_pending_sentences", 4)))
        # 声学级静音用：播报期间在 /voice/speaking 上发布 true，
        # ASR 节点收到后停止录音，避免把机器人自己的话当成用户回答。
        # tail 是播报结束后再多静音一会儿，等房间混响散掉。
        self.speaking_topic = rospy.get_param("~speaking_topic", "/voice/speaking")
        self.speaking_tail_seconds = max(
            0.0, float(rospy.get_param("~speaking_tail_seconds", 0.15))
        )

        self._pending = collections.deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._wav_counter = 0
        self._check_runtime()

        self._worker = threading.Thread(target=self._worker_loop)
        self._worker.daemon = True
        self._worker.start()

        self.sub = rospy.Subscriber(self.say_topic, String, self.say_callback, queue_size=20)
        self.speaking_pub = rospy.Publisher(self.speaking_topic, Bool, queue_size=10)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "Offline TTS ready: topic=%s engine=piper model=%s device=%s",
            self.say_topic,
            self.voice_model,
            self.speaker_device or "system-default",
        )

    def _check_runtime(self):
        if not os.path.exists(self.piper_binary):
            rospy.logerr(
                "Piper binary not found: %s (run tools/setup_offline_voice.sh)", self.piper_binary
            )
        if not os.path.exists(self.voice_model):
            rospy.logerr(
                "Piper voice model not found: %s (run tools/setup_offline_voice.sh)", self.voice_model
            )

    def say_callback(self, msg):
        text = str(msg.data or "").strip()
        if not text:
            return

        with self._lock:
            if len(self._pending) >= self.max_pending:
                dropped = self._pending.popleft()
                rospy.logwarn("TTS queue full, dropping oldest sentence: %s", dropped)
            self._pending.append(text)
        self._wake.set()

    def _worker_loop(self):
        while not rospy.is_shutdown():
            if not self._wake.wait(0.5):
                continue
            with self._lock:
                if not self._pending:
                    self._wake.clear()
                    continue
                text = self._pending.popleft()
            try:
                self.speak(text)
            except Exception as exc:
                rospy.logerr("TTS playback failed: %s", exc)

    def _next_wav_path(self):
        with self._lock:
            self._wav_counter += 1
            return os.path.join(self.tmp_dir, "offline_tts_%05d.wav" % self._wav_counter)

    def synthesize(self, text, wav_path):
        cmd = [self.piper_binary, "--model", self.voice_model, "--output_file", wav_path]
        if self.voice_config:
            cmd.extend(["--config", self.voice_config])
        if abs(self.length_scale - 1.0) > 1e-6:
            cmd.extend(["--length_scale", "%.3f" % self.length_scale])

        start = time.time()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = proc.communicate(input=(text + "\n").encode("utf-8"), timeout=self.synthesis_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("piper synthesis timed out after %.1fs" % self.synthesis_timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                "piper exited with %d: %s"
                % (proc.returncode, stderr.decode("utf-8", errors="replace").strip()[-300:])
            )
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise RuntimeError("piper produced no audio for: %s" % text)
        return time.time() - start

    def play(self, wav_path):
        cmd = [self.player]
        if self.speaker_device:
            cmd.extend(["-D", self.speaker_device])
        cmd.append(wav_path)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            _, stderr = proc.communicate(timeout=self.playback_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("playback timed out after %.1fs" % self.playback_timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                "%s exited with %d: %s"
                % (self.player, proc.returncode, stderr.decode("utf-8", errors="replace").strip()[-300:])
            )

    def speak(self, text):
        wav_path = self._next_wav_path()
        rospy.loginfo("TTS: %s", text)
        # 先举旗：从合成开始就算"正在播报"，这样 ASR 在整句话期间都不录音
        self._publish_speaking(True)
        try:
            synth_seconds = self.synthesize(text, wav_path)
            start = time.time()
            self.play(wav_path)
            rospy.loginfo(
                "TTS done: synthesis %.2fs, playback %.2fs",
                synth_seconds,
                time.time() - start,
            )
        finally:
            # 落旗前多静音一会儿，避免把播报尾音/混响录进去
            if self.speaking_tail_seconds > 0.0:
                time.sleep(self.speaking_tail_seconds)
            self._publish_speaking(False)
            if not self.keep_wav:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _publish_speaking(self, active):
        try:
            self.speaking_pub.publish(Bool(data=bool(active)))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "发布播报状态失败: %s", exc)

    def shutdown(self):
        # 退出时务必落旗，否则 ASR 会一直以为机器人在说话、永远不录音
        self._publish_speaking(False)
        self._wake.set()


def main():
    rospy.init_node("offline_tts_node")
    OfflineTtsNode()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
