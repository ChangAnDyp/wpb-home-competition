#!/usr/bin/env python3
# coding=utf-8

import json
import math
import os
import re
import shutil
import struct
import sys
import threading
import time
import wave

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from perception_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from sound_play.msg import SoundRequest
except Exception:
    SoundRequest = None


def package_dir():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def clamp_int(value, low, high):
    return int(max(low, min(high, value)))


def normalize_vector(vector):
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8 or not np.isfinite(norm):
        return None
    return array / norm


class PersonReidOwnerTest:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_detections = []
        self.latest_detections_stamp = None
        self.extractor = None
        self.reid_backend = "torchreid"
        self.owner_embedding = None
        self.owner_embedding_bank = None
        self.owner_color_embedding = None
        self.owner_profile_meta = {}
        self.match_consecutive_count = 0
        self.last_announce_time = 0.0

        base_dir = package_dir()
        default_profile_dir = os.path.join(base_dir, "data", "reid_owner")
        default_torchreid_root = os.path.join(base_dir, "third_party", "deep-person-reid")

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/perception/person_detections_2d")
        self.say_topic = rospy.get_param("~say_topic", "/voice/say")
        self.sound_topic = rospy.get_param("~sound_topic", "/robotsound")
        self.status_topic = rospy.get_param("~status_topic", "/person_reid_owner_test/status")

        self.profile_dir = os.path.expanduser(rospy.get_param("~profile_dir", default_profile_dir))
        self.profile_path = os.path.expanduser(
            rospy.get_param("~profile_path", os.path.join(self.profile_dir, "owner_profile.npz"))
        )
        self.metadata_path = os.path.expanduser(
            rospy.get_param("~metadata_path", os.path.join(self.profile_dir, "owner_profile.json"))
        )
        self.torchreid_root = os.path.expanduser(rospy.get_param("~torchreid_root", default_torchreid_root))
        self.reid_model_name = rospy.get_param("~reid_model_name", "osnet_x0_25")
        # 重识别权重默认用包里那份"行人重识别"训练权重（Market-1501/MSMT17）。
        # 以前默认是空串 -> torchreid 会退回 ImageNet 预训练权重，那是通用图像
        # 特征、不是用来认人的：实测两位不同的人相似度 0.754，而换成 MSMT17
        # 权重后降到 0.565（区分间隙 0.241 -> 0.432）。
        # 之前只有 launch 文件里设了这个参数，用调试器直接跑脚本时拿不到，
        # 所以一直跑的是 ImageNet 权重 —— 这里把默认值改成正确的权重。
        default_reid_model_path = os.path.join(
            base_dir, "models", "reid", "osnet_x0_25_msmt17.pth"
        )
        self.reid_model_path = os.path.expanduser(
            rospy.get_param("~reid_model_path", default_reid_model_path)
        )
        self.reid_device = rospy.get_param("~reid_device", "cpu")
        self.require_cuda = bool(rospy.get_param("~require_cuda", False))
        self.allow_color_fallback = bool(rospy.get_param("~allow_color_fallback", False))

        self.startup_timeout = float(rospy.get_param("~startup_timeout", 35.0))
        self.image_max_age = float(rospy.get_param("~image_max_age", 1.5))
        self.detections_max_age = float(rospy.get_param("~detections_max_age", 1.5))
        self.detection_min_score = float(rospy.get_param("~detection_min_score", 0.30))
        self.detection_min_area_ratio = float(rospy.get_param("~detection_min_area_ratio", 0.015))
        self.crop_padding = float(rospy.get_param("~crop_padding", 0.08))
        self.crop_top_padding = max(
            self.crop_padding,
            float(rospy.get_param("~crop_top_padding", 0.20)),
        )
        self.crop_bottom_padding = max(
            0.0,
            float(rospy.get_param("~crop_bottom_padding", self.crop_padding)),
        )
        self.top_k_candidates = max(1, int(rospy.get_param("~top_k_candidates", 3)))

        self.reuse_existing_profile = bool(rospy.get_param("~reuse_existing_profile", False))
        self.record_seconds = max(0.5, float(rospy.get_param("~record_seconds", 3.0)))
        self.record_sample_count = max(1, int(rospy.get_param("~record_sample_count", 6)))
        self.record_min_samples = max(1, int(rospy.get_param("~record_min_samples", 3)))
        self.record_sample_interval = max(0.05, float(rospy.get_param("~record_sample_interval", 0.30)))
        self.save_crops = bool(rospy.get_param("~save_crops", True))
        # 文本级回声过滤参数（见 is_likely_echo）。ASR 按窗口切片识别，
        # 机器人自己的提示语要等一个窗口才会变成文字发出来，光靠延时抑制
        # 挡不住，会把自己的话当成用户回答（实测 ③ 报姓名时就踩过）。
        self.asr_echo_guard_seconds = max(
            0.0, float(rospy.get_param("~asr_echo_guard_seconds", 15.0))
        )
        self.asr_echo_min_lcs = max(2, int(rospy.get_param("~asr_echo_min_lcs", 4)))
        self.recent_spoken_texts = []

        self.match_threshold = float(rospy.get_param("~match_threshold", 0.70))
        self.match_required_consecutive = max(1, int(rospy.get_param("~match_required_consecutive", 2)))
        self.match_check_interval = max(0.05, float(rospy.get_param("~match_check_interval", 0.35)))
        self.announce_cooldown = max(0.0, float(rospy.get_param("~announce_cooldown", 5.0)))
        self.stop_after_first_match = bool(rospy.get_param("~stop_after_first_match", False))
        self.save_match_crops = bool(rospy.get_param("~save_match_crops", False))
        self.enable_lying_pose_enhancement = bool(rospy.get_param("~enable_lying_pose_enhancement", True))
        self.lying_aspect_ratio_threshold = max(
            1.0, float(rospy.get_param("~lying_aspect_ratio_threshold", 1.25))
        )
        # 躺姿裁剪：以前是"检测框再外扩 22%"，裁剪图面积是人体框的 1.44 倍，
        # 多出来的全是地面 —— 实测它把 reid 从 0.70 压到 0.60、融合分从 0.626
        # 压到 0.546，8 张躺姿图里 0 张能过线。现在改成：不外扩，反而往里收一点，
        # 让裁剪图贴合人体（实测这一档最优：平均融合分 0.626、5/8 过线）。
        # 注意这里不再用 max(crop_padding, ...) 兜底，否则最低只能到 0.08，收不进去。
        self.lying_crop_padding = max(
            0.0, float(rospy.get_param("~lying_crop_padding", 0.0))
        )
        self.lying_crop_shrink = min(
            0.30, max(0.0, float(rospy.get_param("~lying_crop_shrink", 0.07)))
        )
        self.lying_match_threshold = float(rospy.get_param("~lying_match_threshold", 0.60))
        self.lying_min_reid_score = float(rospy.get_param("~lying_min_reid_score", 0.50))
        self.lying_required_consecutive = max(
            self.match_required_consecutive,
            int(rospy.get_param("~lying_required_consecutive", 3)),
        )
        self.lying_reid_weight = max(0.0, float(rospy.get_param("~lying_reid_weight", 0.75)))
        self.lying_color_weight = max(0.0, float(rospy.get_param("~lying_color_weight", 0.25)))

        self.say_wait_for_subscribers = bool(rospy.get_param("~say_wait_for_subscribers", True))
        self.say_wait_timeout = float(rospy.get_param("~say_wait_timeout", 15.0))
        self.tts_chars_per_second = max(0.1, float(rospy.get_param("~tts_chars_per_second", 6.0)))
        self.tts_min_wait = max(0.0, float(rospy.get_param("~tts_min_wait", 1.0)))
        self.tts_extra_wait = max(0.0, float(rospy.get_param("~tts_extra_wait", 0.4)))
        self.recording_text = rospy.get_param("~recording_text", "正在记录")
        self.record_done_text = rospy.get_param("~record_done_text", "记录结束")
        self.owner_found_text = rospy.get_param("~owner_found_text", "识别到主人")
        self.record_failed_text = rospy.get_param("~record_failed_text", "没有记录到足够的人体画面")
        self.ding_text = rospy.get_param("~ding_text", "叮")
        self.use_sound_play_ding = bool(rospy.get_param("~use_sound_play_ding", True))
        self.ding_frequency_hz = float(rospy.get_param("~ding_frequency_hz", 880.0))
        self.ding_duration = max(0.05, float(rospy.get_param("~ding_duration", 0.18)))
        self.ding_volume = float(rospy.get_param("~ding_volume", 1.0))

        os.makedirs(self.profile_dir, exist_ok=True)

        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=5)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=5, latch=True)
        self.sound_pub = (
            rospy.Publisher(self.sound_topic, SoundRequest, queue_size=2)
            if SoundRequest is not None
            else None
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.detection_sub = rospy.Subscriber(
            self.detections_topic, Detection2DArray, self.detections_callback, queue_size=1
        )

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = time.time()

    def detections_callback(self, message):
        with self.lock:
            self.latest_detections = list(message.detections)
            self.latest_detections_stamp = time.time()

    def publish_status(self, event, **fields):
        payload = {"event": event, "time": time.time()}
        payload.update(fields)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def wait_for_tts(self):
        if not self.say_wait_for_subscribers:
            return
        deadline = time.time() + self.say_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.say_pub.get_num_connections() > 0:
                return
            rospy.sleep(0.05)
        rospy.logwarn("No TTS subscriber connected on %s", self.say_topic)

    def speak(self, text, wait=True):
        rospy.loginfo("TTS: %s", text)
        self.say_pub.publish(String(data=text))
        self.remember_spoken_text(text)
        if wait:
            wait_seconds = max(self.tts_min_wait, len(text) / self.tts_chars_per_second) + self.tts_extra_wait
            rospy.sleep(wait_seconds)

    @staticmethod
    def compact_speech_text(text):
        """去掉空白和标点，只留下用于比对的内容。"""
        compact = re.sub(r"\s+", "", str(text or ""))
        return re.sub(r"[，。！？,.!?、；;：:“”\"'()（）]", "", compact)

    @staticmethod
    def longest_common_substring_length(left, right):
        """两段文本的最长公共子串长度（按字）。文本很短，一维 DP 就够。"""
        text_left = str(left or "")
        text_right = str(right or "")
        if not text_left or not text_right:
            return 0
        previous = [0] * (len(text_right) + 1)
        best = 0
        for left_char in text_left:
            current = [0] * (len(text_right) + 1)
            for index, right_char in enumerate(text_right, start=1):
                if left_char == right_char:
                    current[index] = previous[index - 1] + 1
                    if current[index] > best:
                        best = current[index]
            previous = current
        return best

    ECHO_EXAMPLE_MARKERS = ("例如", "比如", "譬如", "举例")

    @classmethod
    def drop_echo_examples(cls, text):
        """去掉提示语里"例如……"那段，再拿去当回声比对基准。

        为什么要这么做：例子本来就是机器人"期待用户说出口"的内容。
        如果把它也算进回声指纹，用户真的照着例子说一遍（典型场景：
        提示语例子是"姓名张三"、使用者本人就叫张三），就会和自己的
        回声撞成同一个字符串，被 is_likely_echo 丢掉——实测在报姓名
        这一步踩过，表现为"我说了名字，但它一直说超过 30 秒没听到"。
        """
        text = str(text or "")
        for marker in cls.ECHO_EXAMPLE_MARKERS:
            index = text.find(marker)
            if index <= 0:
                continue
            tail = text[index:]
            end = re.search(r"[。！？；!?;]", tail)
            text = text[:index] + (tail[end.end():] if end else "")
        return text

    def remember_spoken_text(self, text):
        """记住机器人刚说过的话，供 is_likely_echo 比对。"""
        compact = self.compact_speech_text(self.drop_echo_examples(text))
        if not compact:
            return
        now = time.time()
        history = list(getattr(self, "recent_spoken_texts", []) or [])
        history.append((now, compact))
        guard = float(getattr(self, "asr_echo_guard_seconds", 15.0))
        keep_after = now - max(1.0, guard * 2.0)
        self.recent_spoken_texts = [
            item for item in history if float(item[0]) >= keep_after
        ][-12:]

    def is_likely_echo(self, text):
        """判断 ASR 文本是不是机器人自己提示语的回声（详见参数注释）。"""
        compact = self.compact_speech_text(text)
        if len(compact) < 4:
            return False
        guard = float(getattr(self, "asr_echo_guard_seconds", 15.0))
        if guard <= 0.0:
            return False
        min_lcs = max(2, int(getattr(self, "asr_echo_min_lcs", 4)))
        now = time.time()
        for spoken_time, spoken in list(getattr(self, "recent_spoken_texts", []) or []):
            if now - float(spoken_time) > guard:
                continue
            if self.longest_common_substring_length(compact, spoken) >= min_lcs:
                return True
        return False

    def ding_wav_path(self):
        return os.path.join(self.profile_dir, "ding.wav")

    def ensure_ding_wav(self):
        path = self.ding_wav_path()
        if os.path.exists(path) and os.path.getsize(path) > 128:
            return path
        sample_rate = 16000
        sample_count = int(sample_rate * self.ding_duration)
        fade_count = max(1, int(sample_rate * min(0.02, self.ding_duration / 4.0)))
        amplitude = int(32767 * max(0.0, min(1.0, self.ding_volume)) * 0.55)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for sample_index in range(sample_count):
                phase = 2.0 * math.pi * self.ding_frequency_hz * sample_index / sample_rate
                envelope = 1.0
                if sample_index < fade_count:
                    envelope = sample_index / float(fade_count)
                elif sample_index > sample_count - fade_count:
                    envelope = max(0.0, (sample_count - sample_index) / float(fade_count))
                value = int(amplitude * envelope * math.sin(phase))
                wav_file.writeframes(struct.pack("<h", value))
        return path

    def play_ding(self):
        if self.use_sound_play_ding and self.sound_pub is not None:
            path = self.ensure_ding_wav()
            request = SoundRequest()
            request.sound = SoundRequest.PLAY_FILE
            request.command = SoundRequest.PLAY_ONCE
            request.volume = max(0.0, min(1.0, self.ding_volume))
            request.arg = path
            self.sound_pub.publish(request)
            rospy.loginfo("Ding played with sound_play: %s", path)
            return
        self.speak(self.ding_text, wait=False)

    def wait_for_camera_inputs(self):
        deadline = time.time() + self.startup_timeout
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                has_image = self.latest_image is not None
                has_detections = self.latest_detections_stamp is not None
            if has_image and has_detections:
                self.publish_status("camera_ready")
                return True
            rate.sleep()
        with self.lock:
            has_image = self.latest_image is not None
            has_detections = self.latest_detections_stamp is not None
        raise RuntimeError(
            "camera/detection inputs not ready: image=%s topic=%s detections=%s topic=%s"
            % (has_image, self.image_topic, has_detections, self.detections_topic)
        )

    def init_reid_backend(self):
        if self.reid_model_path and not os.path.exists(self.reid_model_path):
            raise RuntimeError("reid_model_path does not exist: %s" % self.reid_model_path)
        if os.path.isdir(self.torchreid_root) and self.torchreid_root not in sys.path:
            sys.path.insert(0, self.torchreid_root)
        try:
            import torch
            from torchreid.utils import FeatureExtractor
        except Exception as exc:
            if not self.allow_color_fallback:
                raise RuntimeError(
                    "Torchreid dependencies are not ready. Run: roscd wpb_task1_owner_search && tools/setup_person_reid.sh"
                ) from exc
            rospy.logwarn("Torchreid unavailable; using color-histogram fallback: %s", exc)
            self.reid_backend = "color_hist"
            return
        if self.reid_device.startswith("cuda") and not torch.cuda.is_available():
            if self.require_cuda:
                raise RuntimeError("CUDA requested for Re-ID but torch.cuda.is_available() is false")
            rospy.logwarn("CUDA unavailable for Re-ID; falling back to CPU")
            self.reid_device = "cpu"
        self.extractor = FeatureExtractor(
            model_name=self.reid_model_name,
            model_path=self.reid_model_path,
            device=self.reid_device,
            verbose=False,
        )
        self.reid_backend = "torchreid"
        rospy.loginfo(
            "Person Re-ID ready: backend=torchreid model=%s weights=%s device=%s",
            self.reid_model_name,
            self.reid_model_path or "auto-pretrained",
            self.reid_device,
        )

    def snapshot(self):
        now = time.time()
        with self.lock:
            image = None if self.latest_image is None else self.latest_image.copy()
            image_stamp = self.latest_image_stamp
            detections = list(self.latest_detections)
            detections_stamp = self.latest_detections_stamp
        if image is None or image_stamp is None or now - image_stamp > self.image_max_age:
            return None, []
        if detections_stamp is None or now - detections_stamp > self.detections_max_age:
            return image, []
        return image, detections

    def person_candidates(self, image, detections):
        image_height, image_width = image.shape[:2]
        image_area = float(max(1, image_height * image_width))
        candidates = []
        for detection in detections:
            label = str(getattr(detection, "class_name", "") or "").strip().lower()
            if label and label != "person":
                continue
            score = float(getattr(detection, "score", 0.0) or 0.0)
            if score < self.detection_min_score:
                continue
            xmin = clamp_int(getattr(detection, "xmin", 0), 0, image_width - 1)
            ymin = clamp_int(getattr(detection, "ymin", 0), 0, image_height - 1)
            xmax = clamp_int(getattr(detection, "xmax", 0), 0, image_width - 1)
            ymax = clamp_int(getattr(detection, "ymax", 0), 0, image_height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue
            area_ratio = ((xmax - xmin) * (ymax - ymin)) / image_area
            if area_ratio < self.detection_min_area_ratio:
                continue
            width = xmax - xmin
            height = ymax - ymin
            aspect_ratio = float(width) / float(max(1, height))
            priority = score * area_ratio
            candidates.append(
                {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "score": score,
                    "area_ratio": area_ratio,
                    "aspect_ratio": aspect_ratio,
                    "priority": priority,
                }
            )
        candidates.sort(key=lambda item: item["priority"], reverse=True)
        return candidates[: self.top_k_candidates]

    def crop_candidate(self, image, candidate, padding=None, shrink=None,
                       uniform_padding=False):
        image_height, image_width = image.shape[:2]
        xmin, ymin, xmax, ymax = candidate["bbox"]
        # shrink：先把检测框往内收一点（比例），再按 padding 外扩。
        # 躺姿用它把"检测框本身偏大的那圈背景"也去掉。
        if shrink:
            shrink = min(0.45, max(0.0, float(shrink)))
            shrink_x = (xmax - xmin) * shrink
            shrink_y = (ymax - ymin) * shrink
            xmin += shrink_x
            xmax -= shrink_x
            ymin += shrink_y
            ymax -= shrink_y
        width = xmax - xmin
        height = ymax - ymin
        crop_padding = self.crop_padding if padding is None else float(padding)
        if uniform_padding:
            # 四边用同一个 padding（躺姿用）。站姿那套"上面多留 20%（把头顶算进去）"
            # 对躺着的人没意义 —— 躺着时"上"只是身体的一端，多留只会框进地板。
            top_padding = crop_padding
            bottom_padding = crop_padding
        else:
            top_padding = self.crop_top_padding if padding is None else max(self.crop_top_padding, crop_padding)
            bottom_padding = self.crop_bottom_padding if padding is None else max(self.crop_bottom_padding, crop_padding)
        pad_x = int(width * crop_padding)
        pad_top = int(height * top_padding)
        pad_bottom = int(height * bottom_padding)
        crop_xmin = clamp_int(xmin - pad_x, 0, image_width - 1)
        crop_ymin = clamp_int(ymin - pad_top, 0, image_height - 1)
        crop_xmax = clamp_int(xmax + pad_x, 0, image_width - 1)
        crop_ymax = clamp_int(ymax + pad_bottom, 0, image_height - 1)
        if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
            return None, None
        crop = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
        if crop.size == 0:
            return None, None
        return crop, [crop_xmin, crop_ymin, crop_xmax, crop_ymax]

    def extract_embeddings(self, crops):
        if not crops:
            return []
        if self.reid_backend == "color_hist":
            embeddings = [self.color_hist_embedding(crop) for crop in crops]
            return [embedding for embedding in embeddings if embedding is not None]
        rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]
        features = self.extractor(rgb_crops)
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        embeddings = []
        for feature in np.asarray(features):
            embedding = normalize_vector(feature)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    @staticmethod
    def color_hist_embedding(crop):
        resized = cv2.resize(crop, (96, 192), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        return normalize_vector(hist.reshape(-1))

    def color_similarity(self, crop):
        if self.owner_color_embedding is None:
            return None
        color_embedding = self.color_hist_embedding(crop)
        if color_embedding is None:
            return None
        return float(np.dot(self.owner_color_embedding, color_embedding))

    def is_lying_candidate(self, candidate):
        if not self.enable_lying_pose_enhancement:
            return False
        xmin, ymin, xmax, ymax = candidate["bbox"]
        width = xmax - xmin
        height = ymax - ymin
        aspect_ratio = float(width) / float(max(1, height))
        return aspect_ratio >= self.lying_aspect_ratio_threshold

    def reid_query_variants(self, crop, lying_pose):
        variants = [("raw", crop)]
        if lying_pose and self.enable_lying_pose_enhancement:
            variants.append(("rot90_cw", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)))
            variants.append(("rot90_ccw", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)))
        return variants

    def reid_similarity(self, embedding):
        if self.owner_embedding_bank is not None:
            scores = np.dot(self.owner_embedding_bank, embedding)
            return float(np.max(scores))
        return float(np.dot(self.owner_embedding, embedding))

    def fused_lie_score(self, reid_score, color_score):
        if color_score is None or self.lying_color_weight <= 0.0:
            return reid_score
        total_weight = max(1e-6, self.lying_reid_weight + self.lying_color_weight)
        reid_weight = self.lying_reid_weight / total_weight
        color_weight = self.lying_color_weight / total_weight
        return reid_weight * reid_score + color_weight * color_score

    @staticmethod
    def result_is_match(result, default_threshold):
        if result is None:
            return False
        score = float(result.get("score", -1.0))
        reid_score = float(result.get("reid_score", score))
        threshold = float(result.get("match_threshold", default_threshold))
        min_reid_score = float(result.get("min_reid_score", -1.0))
        return score >= threshold and reid_score >= min_reid_score

    def profile_sample_dir(self):
        # 按主人分目录：samples/owner_1、samples/owner_2 …
        # 这样重新注册某一位时只清掉他自己的样本，不会越攒越多（实测以前
        # 文件名带毫秒时间戳、又全堆在一个目录，注册几次就是几百个文件）。
        sample_dir = os.path.join(
            self.profile_dir, "samples", "owner_%d" % max(1, int(getattr(self, "current_owner_index", 1) or 1))
        )
        os.makedirs(sample_dir, exist_ok=True)
        return sample_dir

    def match_sample_dir(self):
        match_dir = os.path.join(
            self.profile_dir, "matches", "owner_%d" % max(1, int(getattr(self, "current_owner_index", 1) or 1))
        )
        os.makedirs(match_dir, exist_ok=True)
        return match_dir

    def reset_owner_sample_dirs(self, owner_index):
        """重新注册某一位主人前，先清掉他上一次的样本图，避免无限堆积。

        每位主人最多留一份（本次注册的）样本；档案本体（.npz/.json）本来就是
        按 owner_<N> 覆盖写入的，所以清理样本后就完全不会累积了。
        """
        index = max(1, int(owner_index or 1))
        for name in ("samples", "matches"):
            path = os.path.join(self.profile_dir, name, "owner_%d" % index)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    def owner_profile_paths(self, owner_index):
        """某一位主人的档案路径（基类是单档案；多主人版会覆盖这个方法）。"""
        return self.profile_path, self.metadata_path

    def delete_owner_profile(self, owner_index):
        """删掉某一位主人的档案（本轮被"跳过"时用）。

        这样"新一轮注册"之后，磁盘上的档案集合 = 本轮真正注册成功的人，
        不会残留上一次的旧主人信息被后续误用。
        """
        profile_path, metadata_path = self.owner_profile_paths(owner_index)
        removed = []
        for path in (profile_path, metadata_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    removed.append(path)
                except OSError as exc:
                    rospy.logwarn("删除旧档案失败 %s：%s", path, exc)
        self.reset_owner_sample_dirs(owner_index)
        if removed:
            rospy.loginfo(
                "第 %s 位主人被跳过 -> 已删除原有档案：%s",
                owner_index,
                "、".join(os.path.basename(p) for p in removed),
            )
        return removed

    def save_crop(self, crop, directory, prefix, index):
        if not self.save_crops:
            return ""
        filename = "%s_%03d_%d.jpg" % (prefix, index, int(time.time() * 1000))
        path = os.path.join(directory, filename)
        cv2.imwrite(path, crop)
        return path

    def save_owner_profile(self, embeddings, sample_meta, color_embeddings=None):
        mean_embedding = normalize_vector(np.mean(np.vstack(embeddings), axis=0))
        if mean_embedding is None:
            raise RuntimeError("owner embedding is invalid")
        embedding_bank = [mean_embedding]
        for embedding in embeddings:
            normalized = normalize_vector(embedding)
            if normalized is not None:
                embedding_bank.append(normalized)
        embedding_bank = np.vstack(embedding_bank).astype(np.float32)
        npz_payload = {
            "embedding": mean_embedding.astype(np.float32),
            "embedding_bank": embedding_bank,
        }
        mean_color_embedding = None
        if color_embeddings:
            mean_color_embedding = normalize_vector(np.mean(np.vstack(color_embeddings), axis=0))
            if mean_color_embedding is not None:
                npz_payload["color_embedding"] = mean_color_embedding.astype(np.float32)
        np.savez(self.profile_path, **npz_payload)
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": self.reid_backend,
            "model_name": self.reid_model_name,
            "model_path": self.reid_model_path,
            "device": self.reid_device,
            "match_threshold": self.match_threshold,
            "lying_pose_enhancement": self.enable_lying_pose_enhancement,
            "lying_match_threshold": self.lying_match_threshold,
            "lying_min_reid_score": self.lying_min_reid_score,
            "lying_required_consecutive": self.lying_required_consecutive,
            "has_color_embedding": mean_color_embedding is not None,
            "samples": sample_meta,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        self.owner_embedding = mean_embedding
        self.owner_embedding_bank = embedding_bank
        self.owner_color_embedding = mean_color_embedding
        self.owner_profile_meta = metadata
        self.publish_status("profile_saved", profile_path=self.profile_path, samples=len(sample_meta))

    def load_owner_profile(self):
        if not os.path.exists(self.profile_path):
            return False
        profile = np.load(self.profile_path, allow_pickle=False)
        embedding = normalize_vector(profile["embedding"])
        if embedding is None:
            raise RuntimeError("owner profile exists but embedding is invalid: %s" % self.profile_path)
        self.owner_embedding = embedding
        bank = []
        if "embedding_bank" in profile.files:
            for vector in np.asarray(profile["embedding_bank"]):
                normalized = normalize_vector(vector)
                if normalized is not None:
                    bank.append(normalized)
        if not bank:
            bank.append(embedding)
        self.owner_embedding_bank = np.vstack(bank).astype(np.float32)
        self.owner_color_embedding = None
        if "color_embedding" in profile.files:
            self.owner_color_embedding = normalize_vector(profile["color_embedding"])
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as metadata_file:
                self.owner_profile_meta = json.load(metadata_file)
        self.warn_if_profile_model_mismatch()
        self.publish_status("profile_loaded", profile_path=self.profile_path)
        return True

    def warn_if_profile_model_mismatch(self):
        """档案里的重识别权重和当前用的不一致时告警。

        不同权重算出来的 embedding 不在同一个空间里，硬比出来的相似度没有意义
        （实测：ImageNet 权重下两个人 0.754，换成 MSMT17 权重后 0.565）。
        所以换了权重就必须重新注册 —— 这里至少让它明确报警，而不是悄悄地认不准。
        """
        recorded = str(self.owner_profile_meta.get("model_path", "") or "")
        current = str(self.reid_model_path or "")
        if recorded == current:
            return
        rospy.logwarn(
            "档案 %s 是用「%s」注册的，和当前用的「%s」不一致 —— "
            "两者算出的特征不可比，请重新注册该主人",
            os.path.basename(self.profile_path),
            os.path.basename(recorded) if recorded else "ImageNet 预训练权重",
            os.path.basename(current) if current else "ImageNet 预训练权重",
        )
        self.publish_status(
            "profile_model_mismatch",
            profile=os.path.basename(self.profile_path),
            recorded=recorded,
            current=current,
        )

    def record_owner(self):
        self.publish_status("recording_started")
        self.speak(self.recording_text, wait=True)
        self.play_ding()
        embeddings = []
        color_embeddings = []
        sample_meta = []
        sample_dir = self.profile_sample_dir()
        deadline = time.time() + self.record_seconds
        next_sample_time = 0.0
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and time.time() < deadline and len(embeddings) < self.record_sample_count:
            now = time.time()
            if now < next_sample_time:
                rate.sleep()
                continue
            next_sample_time = now + self.record_sample_interval
            image, detections = self.snapshot()
            if image is None:
                rospy.logwarn_throttle(1.0, "Waiting for fresh camera image on %s", self.image_topic)
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if not candidates:
                rospy.logwarn_throttle(1.0, "Waiting for person detection on %s", self.detections_topic)
                rate.sleep()
                continue
            crop, crop_bbox = self.crop_candidate(image, candidates[0])
            if crop is None:
                rate.sleep()
                continue
            extracted = self.extract_embeddings([crop])
            if not extracted:
                rate.sleep()
                continue
            sample_index = len(embeddings) + 1
            path = self.save_crop(crop, sample_dir, "owner", sample_index)
            color_embedding = self.color_hist_embedding(crop)
            if color_embedding is not None:
                color_embeddings.append(color_embedding)
            embeddings.append(extracted[0])
            sample_meta.append(
                {
                    "index": sample_index,
                    "image": path,
                    "bbox": candidates[0]["bbox"],
                    "crop_bbox": crop_bbox,
                    "det_score": candidates[0]["score"],
                    "area_ratio": candidates[0]["area_ratio"],
                }
            )
            self.publish_status("recording_sample", sample=sample_index, required=self.record_min_samples)
            rospy.loginfo("Owner Re-ID sample %d captured: %s", sample_index, path or "not saved")
            rate.sleep()
        if len(embeddings) < self.record_min_samples:
            self.publish_status("recording_failed", samples=len(embeddings), required=self.record_min_samples)
            self.speak(self.record_failed_text, wait=True)
            raise RuntimeError("only captured %d/%d usable Re-ID samples" % (len(embeddings), self.record_min_samples))
        self.save_owner_profile(embeddings, sample_meta, color_embeddings=color_embeddings)
        self.speak(self.record_done_text, wait=True)
        self.publish_status("recording_done", samples=len(embeddings))

    def evaluate_current_frame(self):
        image, detections = self.snapshot()
        if image is None:
            return None
        candidates = self.person_candidates(image, detections)
        if not candidates:
            return None
        query_crops = []
        records = []
        for candidate in candidates:
            lying_pose = self.is_lying_candidate(candidate)
            padding = self.lying_crop_padding if lying_pose else self.crop_padding
            crop, crop_bbox = self.crop_candidate(image, candidate, padding=padding)
            if crop is None:
                continue
            meta = dict(candidate)
            meta["crop_bbox"] = crop_bbox
            meta["lying_pose"] = lying_pose
            variants = self.reid_query_variants(crop, lying_pose)
            query_indexes = []
            variant_names = []
            for variant_name, variant_crop in variants:
                query_indexes.append(len(query_crops))
                variant_names.append(variant_name)
                query_crops.append(variant_crop)
            records.append(
                {
                    "crop": crop,
                    "meta": meta,
                    "query_indexes": query_indexes,
                    "variant_names": variant_names,
                }
            )
        if not query_crops:
            return None
        embeddings = self.extract_embeddings(query_crops)
        if len(embeddings) != len(query_crops):
            return None
        best_result = None
        for record in records:
            variant_scores = []
            for query_index, variant_name in zip(record["query_indexes"], record["variant_names"]):
                variant_scores.append((self.reid_similarity(embeddings[query_index]), variant_name))
            reid_score, best_variant = max(variant_scores, key=lambda item: item[0])
            color_score = self.color_similarity(record["crop"])
            lying_pose = record["meta"].get("lying_pose", False)
            if lying_pose:
                score = self.fused_lie_score(reid_score, color_score)
                match_threshold = self.lying_match_threshold
                min_reid_score = self.lying_min_reid_score
                required_consecutive = self.lying_required_consecutive
            else:
                score = reid_score
                match_threshold = self.match_threshold
                min_reid_score = -1.0
                required_consecutive = self.match_required_consecutive
            result = {
                "score": float(score),
                "reid_score": float(reid_score),
                "color_score": None if color_score is None else float(color_score),
                "match_threshold": match_threshold,
                "min_reid_score": min_reid_score,
                "required_consecutive": required_consecutive,
                "best_variant": best_variant,
                "candidate": record["meta"],
                "crop": record["crop"],
                "num_candidates": len(records),
            }
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
        return best_result

    def maybe_save_match_crop(self, crop, score):
        if not self.save_match_crops:
            return ""
        return self.save_crop(crop, self.match_sample_dir(), "match_%.2f" % score, 1)

    def recognition_loop(self):
        self.publish_status("recognition_started", threshold=self.match_threshold)
        rate = rospy.Rate(max(1.0, 1.0 / self.match_check_interval))
        while not rospy.is_shutdown():
            result = self.evaluate_current_frame()
            if result is None:
                self.match_consecutive_count = 0
                rate.sleep()
                continue
            score = result["score"]
            match_threshold = result.get("match_threshold", self.match_threshold)
            required_consecutive = result.get("required_consecutive", self.match_required_consecutive)
            matched = self.result_is_match(result, self.match_threshold)
            if matched:
                self.match_consecutive_count += 1
            else:
                self.match_consecutive_count = 0
            self.publish_status(
                "match_score",
                score=score,
                reid_score=result.get("reid_score", score),
                color_score=result.get("color_score"),
                matched=matched,
                consecutive=self.match_consecutive_count,
                threshold=match_threshold,
                required_consecutive=required_consecutive,
                lying_pose=result.get("candidate", {}).get("lying_pose", False),
                best_variant=result.get("best_variant", "raw"),
            )
            if self.match_consecutive_count >= required_consecutive:
                now = time.time()
                if now - self.last_announce_time >= self.announce_cooldown:
                    crop_path = self.maybe_save_match_crop(result["crop"], score)
                    rospy.loginfo("Owner recognized: score=%.3f crop=%s", score, crop_path or "not saved")
                    self.speak(self.owner_found_text, wait=False)
                    self.publish_status("owner_recognized", score=score, crop=crop_path)
                    self.last_announce_time = now
                    if self.stop_after_first_match:
                        return
            rate.sleep()

    def run(self):
        self.wait_for_tts()
        self.wait_for_camera_inputs()
        self.init_reid_backend()
        if self.reuse_existing_profile and self.load_owner_profile():
            rospy.loginfo("Loaded existing owner Re-ID profile: %s", self.profile_path)
        else:
            self.record_owner()
        if self.owner_embedding is None:
            raise RuntimeError("owner profile is not ready")
        self.recognition_loop()


def main():
    rospy.init_node("person_reid_owner_test")
    node = PersonReidOwnerTest()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("Person Re-ID owner test failed: %s", exc)
        node.publish_status("error", message=str(exc))
        raise


if __name__ == "__main__":
    main()
