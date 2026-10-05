#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS1 local YOLO Pose action recognition node.

The node keeps the latest camera frame for display, warms up the local
vision model once, and analyzes a short sequence of camera frames. Press
SPACE to recognize again; press Q or ESC to quit the preview.
"""

import json
import math
import os
import re
import threading
import time

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String


DEFAULT_IMAGE_TOPIC = "/kinect2/qhd/image_color_rect"
DEFAULT_POINTS_TOPIC = "/kinect2/qhd/points"
DEFAULT_SAY_TOPIC = "/voice/say"
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_POSE_MODEL_CANDIDATES = (
    os.path.join(
        WORKSPACE_ROOT,
        "src",
        "wpb_task1_owner_search",
        "models",
        "pose",
        "yolo11n-pose.pt",
    ),
    os.path.join(WORKSPACE_ROOT, "yolo11n-pose.pt"),
    os.path.join(os.path.expanduser("~"), "models", "yolo11n-pose.pt"),
)

ACTION_UNKNOWN = "unknown"
PLACE_UNKNOWN = "unknown"

PLACE_TEXT = {
    "chair": "椅子上",
    "sofa": "沙发上",
    "bed": "床上",
    "floor": "地上",
    "none": "",
    "unknown": "",
}

# 预览窗口叠加用的中文标签（英文结论 -> 中文显示）
ACTION_TEXT = {
    "sitting": "坐着",
    "lying": "躺着",
    "fallen": "摔倒在地",
    "sudden_fall": "突然摔倒",
    "waving": "挥手",
    "unknown": "未知",
}

PLACE_LABEL = {
    "chair": "椅子",
    "sofa": "沙发",
    "bed": "床",
    "floor": "地面",
    "none": "",
    "unknown": "未知",
}


def draw_text_box(image, text, origin, color, scale=0.6, thickness=2):
    """在图像上画一行带黑底的文字，返回下一行建议的 y 坐标。"""
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(
        image,
        (max(0, x - 4), max(0, y - text_h - baseline - 2)),
        (min(image.shape[1] - 1, x + text_w + 4), y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return y + text_h + baseline + 10


# ---------------------------------------------------------------------------
# 本地补写（本机适配）：PROMPT
#
# 背景：本文件是从作者处取得的**纯姿态版本**，模块里没有任何 LLM 相关内容。
# 而 task1_find_owner_real.py 在没显式传 prompt 时会取
# self.qwen_action_core.PROMPT 作为默认提示词，缺它会在运行时抛
# AttributeError（prompt 为 None 时 `prompt or PROMPT` 会求值到它）。
#
# 下面的提示词按本文件 normalize_result()/parse_result() 实际能识别的
# 词表编写，保证模型输出的取值能被正确解析：
#   action: sitting / lying / fallen / sudden_fall / waving / unknown
#   place : chair / sofa / bed / floor / unknown
# 开头 /no_think 与任务节点里 warmup_prompt 的写法保持一致。
# ---------------------------------------------------------------------------
PROMPT = (
    "/no_think\n"
    "你是家庭服务机器人的视觉模块。请观察这几张连续画面，判断画面里那个人的动作。\n"
    "只输出一行 JSON，不要解释，不要输出多余文字，不要换行。\n"
    'action 只能取以下之一：\n'
    "  sitting      - 坐着\n"
    "  lying        - 躺/卧在沙发、床或椅子上\n"
    "  fallen       - 已经摔倒并躺在地面上\n"
    "  sudden_fall  - 正在从站立倒向地面（画面里能看到倒地过程）\n"
    "  waving       - 正在挥手示意\n"
    "  unknown      - 看不清或无法判断\n"
    'place 只能取以下之一：chair / sofa / bed / floor / unknown\n'
    "判断依据：人在沙发上就填 sofa，在床上填 bed，坐在或躺在椅子上填 chair，"
    "在地面上填 floor，判断不出填 unknown。\n"
    '输出格式：{"action":"...","place":"..."}\n'
    "注意：静止躺在地面应填 fallen；躺在沙发、床或椅子等家具上应填 lying；"
    "只有明确看到从站立到倒地的过程才填 sudden_fall。"
)


def normalize_value(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_owner_roi(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = re.split(r"[,;\s]+", str(value).strip())
    if len(parts) < 4:
        return None
    try:
        roi = tuple(int(round(float(item))) for item in parts[:4])
    except (TypeError, ValueError):
        return None
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        return None
    return roi


def normalize_result(parsed):
    if not isinstance(parsed, dict):
        return ACTION_UNKNOWN, PLACE_UNKNOWN

    action_raw = normalize_value(parsed.get("action"))
    place_raw = normalize_value(parsed.get("place"))

    action_aliases = {
        "坐": "sitting",
        "坐着": "sitting",
        "坐下": "sitting",
        "sitting": "sitting",
        "sit": "sitting",
        "躺": "lying",
        "躺着": "lying",
        "躺下": "lying",
        "lying": "lying",
        "lie": "lying",
        "摔倒": "fallen",
        "倒地": "fallen",
        "跌倒": "fallen",
        "fallen": "fallen",
        "fall": "fallen",
        "突然摔倒": "sudden_fall",
        "突然倒地": "sudden_fall",
        "sudden_fall": "sudden_fall",
        "sudden fall": "sudden_fall",
        "falling": "sudden_fall",
        "挥手": "waving",
        "挥手示意": "waving",
        "waving": "waving",
        "wave": "waving",
        "未知": ACTION_UNKNOWN,
        "无法判断": ACTION_UNKNOWN,
        "unknown": ACTION_UNKNOWN,
        "": ACTION_UNKNOWN,
    }
    place_aliases = {
        "椅子": "chair",
        "椅子上": "chair",
        "chair": "chair",
        "沙发": "sofa",
        "沙发上": "sofa",
        "sofa": "sofa",
        "床": "bed",
        "床上": "bed",
        "bed": "bed",
        "地面": "floor",
        "地板": "floor",
        "地板上": "floor",
        "地面上": "floor",
        "地上": "floor",
        "floor": "floor",
        "none": "none",
        "无": "none",
        "不适用": "none",
        "未知": PLACE_UNKNOWN,
        "unknown": PLACE_UNKNOWN,
        "": PLACE_UNKNOWN,
    }

    action = action_aliases.get(action_raw, ACTION_UNKNOWN)
    place = place_aliases.get(place_raw, PLACE_UNKNOWN)

    if action == ACTION_UNKNOWN:
        parsed_text = normalize_value(json.dumps(parsed, ensure_ascii=False))
        if any(token in parsed_text for token in ("突然摔倒", "突然倒地", "sudden_fall", "sudden fall", "falling")):
            action = "sudden_fall"
        elif any(token in parsed_text for token in ("摔倒", "跌倒", "倒地", "fallen")):
            action = "fallen"
        elif any(token in parsed_text for token in ("挥手", "waving", "wave")):
            action = "waving"
        elif (
            any(token in parsed_text for token in ("躺", "lying", "lie"))
            and any(token in parsed_text for token in ("地上", "地面", "地板", "floor"))
        ):
            action = "fallen"
    if action in ("fallen", "sudden_fall"):
        place = "floor"
    if action == "lying" and place == "floor":
        action = "fallen"
    if action == ACTION_UNKNOWN:
        place = PLACE_UNKNOWN
    return action, place


def parse_result(content):
    text = (content or "").strip()
    try:
        parsed = json.loads(text)
        return normalize_result(parsed)
    except (TypeError, json.JSONDecodeError):
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return normalize_result(json.loads(match.group(0)))
        except (TypeError, json.JSONDecodeError):
            pass

    lowered = text.lower()
    if (
        "突然摔倒" in text
        or "突然跌倒" in text
        or "突然倒地" in text
        or "sudden_fall" in lowered
        or "sudden fall" in lowered
        or "falling" in lowered
    ):
        return "sudden_fall", "floor"
    if "摔倒" in text or "跌倒" in text or "倒地" in text or "fallen" in lowered:
        return "fallen", "floor"
    if "挥手" in text or "waving" in lowered or "wave" in lowered:
        return "waving", PLACE_UNKNOWN
    if "躺" in text or "lying" in lowered or "lie" in lowered:
        if "地上" in text or "地面" in text or "地板" in text or "floor" in lowered:
            return "fallen", "floor"
        if "沙发" in text or "sofa" in lowered:
            return "lying", "sofa"
        if "床" in text or "bed" in lowered:
            return "lying", "bed"
        if "椅" in text or "chair" in lowered:
            return "lying", "chair"
        return "lying", PLACE_UNKNOWN
    if "坐" in text or "sitting" in lowered or "sit" in lowered:
        if "沙发" in text or "sofa" in lowered:
            return "sitting", "sofa"
        if "床" in text or "bed" in lowered:
            return "sitting", "bed"
        if "椅" in text or "chair" in lowered:
            return "sitting", "chair"
        return "sitting", PLACE_UNKNOWN
    return ACTION_UNKNOWN, PLACE_UNKNOWN


def result_to_speech(action, place):
    if action == "sudden_fall":
        return "主人突然摔倒在地上。"
    if action == "fallen":
        return "主人已经摔倒在地上。"
    if action == "waving":
        if place in ("chair", "sofa", "bed"):
            return "主人正在挥手，人在%s。" % PLACE_TEXT[place]
        return "主人正在挥手。"
    if action == "sitting":
        if place in ("chair", "sofa", "bed"):
            return "主人正坐在%s。" % PLACE_TEXT[place]
        return "主人正在坐着。"
    if action == "lying":
        if place in ("chair", "sofa", "bed"):
            return "主人正躺在%s。" % PLACE_TEXT[place]
        return "主人正在躺着。"
    return "暂时无法判断主人的动作。"


def final_posture_is_horizontal(features):
    """判断"最终是不是横躺"，用最后一段帧的比例而不是只看最后一帧。

    姿态分类器判定"倒下 / 已经躺下"时，用的是最后 1/4 帧里 lying_like 的比例
    （见 PoseActionAnalyzer.classify）。高度判断必须用同一口径，否则
    "从坐到躺"的过程中最后一帧还没完全躺平，高度判断就被跳过，
    躺在床上也会被判成 sudden_fall（主人摔倒）。
    """
    if not features:
        return False
    split = max(2, int(math.ceil(len(features) / 4.0)))
    last = features[-split:]
    # 这里只回答一个问题："人最后是不是横躺下来了"。
    # 注意不要跟着 classify 的 final_posture_is_low 把 sitting_like 也算进来——
    # 那样"坐在床上"也会被判成姿态放低，进而触发 elevated_lying -> lying，
    # 把坐着的人说成躺着。
    # "坐在床上往下躺"最后会变成 lying_like，所以只看横躺就够了；
    # 万一最后一帧还没躺平，也由 merge 里"家具面 + sudden_fall -> lying"
    # 那条规则兜住，不依赖这里。
    lying = sum(1 for feature in last if feature.get("lying_like"))
    return (lying / float(len(last))) >= 0.30


def relation_is_elevated(ground_relation):
    """判断支撑面是否高于地面（床/沙发/椅等家具面）。

    高度优先级：
      1) 点云直接给出 status == elevated
      2) 支撑面绝对高度中位数 >= 家具高度阈值（默认 0.42m）
      3) 相对本地地面的抬升量 >= 0.24m（兜底，本地地面不可靠时仅供参考）

    高度单位是“距地高度（米）”，由 PointCloudGroundAnalyzer 用相机安装高度
    换算得到，所以 pointcloud_camera_height 必须接近真实相机离地高度。
    """
    if not ground_relation:
        return False
    if ground_relation.get("status") == "elevated":
        return True
    median = ground_relation.get("surface_height_median")
    limit = ground_relation.get("furniture_height_limit")
    if median is not None and limit is not None:
        return float(median) >= float(limit)
    delta = ground_relation.get("height_above_local_floor")
    if delta is not None:
        return float(delta) >= 0.24
    return False


def merge_pose_action_result(pose_action, ground_relation, final_pose_is_horizontal):
    ground_relation = ground_relation or {}
    elevated_surface = relation_is_elevated(ground_relation)
    elevated_lying = bool(
        elevated_surface and final_pose_is_horizontal
    )
    ground_fallen = bool(
        ground_relation.get("ground_like")
        and ground_relation.get("ground_confident", True)
        and (
            final_pose_is_horizontal
            or pose_action in ("lying", "sudden_fall", "fallen")
        )
    )

    # 高度优先：人最终横躺在家具面（床/沙发/椅）上时，即使姿态检测到“倒下过程”，
    # 也应判为 lying，而不是 fallen。
    if elevated_lying and pose_action in ("sudden_fall", "fallen", "lying"):
        return "lying", PLACE_UNKNOWN, ground_fallen, True
    # 支撑面已经是家具面时，不可能"摔倒在地"——那只是往家具上躺下去。
    # （高度算准之后，这条才敢放开：地上躺着的读数在任何距离都接近 0。）
    if elevated_surface and pose_action == "sudden_fall":
        return "lying", PLACE_UNKNOWN, ground_fallen, True
    if pose_action == "sudden_fall":
        return "sudden_fall", "floor", ground_fallen, elevated_lying
    if ground_fallen:
        return "fallen", "floor", ground_fallen, elevated_lying
    if elevated_lying:
        return "lying", PLACE_UNKNOWN, ground_fallen, elevated_lying
    if pose_action == "waving":
        return "waving", PLACE_UNKNOWN, ground_fallen, elevated_lying
    if pose_action in ("sitting", "lying"):
        return pose_action, PLACE_UNKNOWN, ground_fallen, elevated_lying
    if pose_action == "fallen":
        return "fallen", "floor", ground_fallen, elevated_lying
    return ACTION_UNKNOWN, PLACE_UNKNOWN, ground_fallen, elevated_lying


# ---------------------------------------------------------------------------
# 本地补写（本机适配）：merge_action_result
#
# 背景：本文件是从作者处取得的**纯姿态版本**，只带 3 参数的
# merge_pose_action_result。而 wpb_task1_owner_search 的
# task1_find_owner_real.py 在静态动作分支里调用的是 5 参数的
# merge_action_result(qwen_action, place, pose_action, ground_relation,
#                     final_pose_is_horizontal)
# 缺少它会让动作识别在运行时抛 AttributeError。
#
# 这里按任务节点的调用点、以及同文件 merge_pose_action_result 的既有判据
# 重新实现：地面/家具抬升的判定逻辑与上面完全一致，只是把“Qwen 静态结论”
# 也纳入决策。判据优先级：
#   1) 姿态给出的 sudden_fall / waving（动态动作必须有姿态证据）
#   2) 点云判定为地面 -> fallen
#   3) 点云判定为家具抬升且 Qwen 说躺着/看不清 -> lying
#   4) Qwen 的静态结论（sitting / lying / fallen / waving）
#   5) 姿态的静态结论（sitting / lying）
#   6) unknown
#
# 因为作者原版拿不到，这段属于**行为等价推断**，不是原版逐行拷贝；
# 若日后拿到原版文件，可用原版整体替换本文件。
# ---------------------------------------------------------------------------
_MERGE_ACTIONS = ("sitting", "lying", "fallen", "sudden_fall", "waving")


def merge_action_result(
    qwen_action,
    place,
    pose_action,
    ground_relation,
    final_pose_is_horizontal,
):
    """融合 Qwen 静态结论、YOLO-Pose 结论与点云支撑面分析。

    返回 (action, place, ground_fallen, elevated_lying)，
    与 task1_find_owner_real.py 的期望完全一致。
    """
    ground_relation = ground_relation or {}
    qwen_action = normalize_value(qwen_action) or ACTION_UNKNOWN
    qwen_place = normalize_value(place) or PLACE_UNKNOWN
    pose_action = normalize_value(pose_action) or ACTION_UNKNOWN
    if qwen_action not in _MERGE_ACTIONS:
        qwen_action = ACTION_UNKNOWN
    if pose_action not in _MERGE_ACTIONS:
        pose_action = ACTION_UNKNOWN

    # 与 merge_pose_action_result 保持一致的几何判据
    elevated_surface = relation_is_elevated(ground_relation)
    elevated_lying = bool(
        elevated_surface and final_pose_is_horizontal
    )
    ground_fallen = bool(
        ground_relation.get("ground_like")
        and ground_relation.get("ground_confident", True)
        and (
            final_pose_is_horizontal
            or pose_action in ("lying", "sudden_fall", "fallen")
            or qwen_action in ("lying", "fallen", "sudden_fall")
        )
    )

    # 0) 高度优先：人最终横躺在家具面（床/沙发/椅）上时，无论姿态还是 Qwen
    #    给出“倒下/摔倒”，都应判为 lying 而不是 fallen。
    if elevated_lying and (
        pose_action in ("sudden_fall", "fallen", "lying")
        or qwen_action in ("fallen", "sudden_fall", "lying")
    ):
        return "lying", PLACE_UNKNOWN, ground_fallen, True
    # 家具面上不可能"摔倒在地"：姿态报 sudden_fall 也只是往家具上躺
    if elevated_surface and pose_action == "sudden_fall" and not ground_fallen:
        return "lying", PLACE_UNKNOWN, ground_fallen, True

    # 1) 动态动作只认姿态证据
    if pose_action == "sudden_fall":
        return "sudden_fall", "floor", ground_fallen, elevated_lying
    if pose_action == "waving":
        return "waving", qwen_place, ground_fallen, elevated_lying
    if qwen_action in ("sudden_fall", "waving"):
        # 任务节点已在外层过滤过一次，这里再兜一层，避免只凭图片就报动态动作
        qwen_action = ACTION_UNKNOWN

    # 2) 点云确认在地面上 -> 摔倒
    if ground_fallen:
        return "fallen", "floor", ground_fallen, elevated_lying

    # 3) 点云确认支撑面是家具且人是横躺的 -> 躺在家具上
    if elevated_lying and qwen_action in ("lying", ACTION_UNKNOWN):
        return "lying", PLACE_UNKNOWN, ground_fallen, elevated_lying

    # 4) 信任 Qwen 的静态结论
    if qwen_action in ("sitting", "lying", "fallen"):
        merged_place = qwen_place if qwen_place not in ("", PLACE_UNKNOWN) else PLACE_UNKNOWN
        if qwen_action == "fallen":
            merged_place = "floor"
        return qwen_action, merged_place, ground_fallen, elevated_lying

    # 5) 退到姿态的静态结论
    if pose_action in ("sitting", "lying", "fallen"):
        merged_place = "floor" if pose_action == "fallen" else PLACE_UNKNOWN
        return pose_action, merged_place, ground_fallen, elevated_lying

    return ACTION_UNKNOWN, PLACE_UNKNOWN, ground_fallen, elevated_lying


class PoseActionAnalyzer:
    def __init__(
        self,
        model_path,
        device,
        image_size,
        confidence,
        iou,
        max_detections,
        half=False,
    ):
        self.model_path = model_path
        self.device = device
        self.half = bool(half)
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self.model = None
        self.ready = False

    def _resolve_device(self):
        """请求 CUDA 但不可用时回退到 CPU，避免整个节点启动失败。"""
        device = str(self.device or "cpu")
        if not device.startswith("cuda"):
            return device, ""
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu", "请求 %s 但 torch.cuda.is_available()=False，已回退 CPU" % device
        except Exception as exc:
            return "cpu", "请求 %s 但检查 CUDA 失败(%s)，已回退 CPU" % (device, exc)
        return device, ""

    @staticmethod
    def resolve_model_path(configured_path):
        if configured_path:
            path = os.path.expanduser(configured_path)
            return path if os.path.exists(path) else ""
        for path in DEFAULT_POSE_MODEL_CANDIDATES:
            if os.path.exists(path):
                return path
        return ""

    def initialize(self):
        if not self.model_path:
            return False, "pose model file not found"
        device, note = self._resolve_device()
        if note:
            rospy.logwarn("Pose device: %s", note)
            self.device = device
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.ready = True
            return True, "pose model loaded on %s" % self.device
        except Exception as exc:
            self.model = None
            self.ready = False
            return False, str(exc)

    @staticmethod
    def _tensor_to_list(value):
        if value is None:
            return []
        try:
            return value.cpu().numpy()
        except Exception:
            return value

    @staticmethod
    def _point(keypoints, confidences, index, minimum_confidence):
        if index >= len(keypoints) or index >= len(confidences):
            return None
        if float(confidences[index]) < minimum_confidence:
            return None
        return float(keypoints[index][0]), float(keypoints[index][1])

    @staticmethod
    def _midpoint(first, second):
        if first is None or second is None:
            return None
        return (0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1]))

    @staticmethod
    def _angle(first, middle, last):
        if first is None or middle is None or last is None:
            return None
        first_vector = (first[0] - middle[0], first[1] - middle[1])
        last_vector = (last[0] - middle[0], last[1] - middle[1])
        first_length = math.hypot(*first_vector)
        last_length = math.hypot(*last_vector)
        if first_length <= 1e-6 or last_length <= 1e-6:
            return None
        cosine = (
            first_vector[0] * last_vector[0]
            + first_vector[1] * last_vector[1]
        ) / (first_length * last_length)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    def _select_person(self, result, width, target_center=None):
        if result is None or result.keypoints is None or result.boxes is None:
            return None
        try:
            keypoints = self._tensor_to_list(result.keypoints.xy)
            keypoint_confidences = self._tensor_to_list(result.keypoints.conf)
            boxes = self._tensor_to_list(result.boxes.xyxy)
            box_confidences = self._tensor_to_list(result.boxes.conf)
        except Exception:
            return None
        count = min(len(keypoints), len(keypoint_confidences), len(boxes), len(box_confidences))
        best = None
        best_score = -1.0
        target_center = 0.5 if target_center is None else float(target_center)
        for index in range(count):
            valid_keypoints = sum(
                1 for value in keypoint_confidences[index]
                if float(value) >= 0.20
            )
            if valid_keypoints < 5:
                continue
            box = boxes[index]
            center = 0.5 * (float(box[0]) + float(box[2])) / max(1.0, float(width))
            score = (
                float(box_confidences[index])
                + 0.03 * valid_keypoints
                - abs(center - target_center)
            )
            if score > best_score:
                best_score = score
                best = {
                    "keypoints": keypoints[index],
                    "confidences": keypoint_confidences[index],
                    "box": box,
                    "box_confidence": float(box_confidences[index]),
                }
        return best

    def _feature(self, pose, image_shape):
        height, width = image_shape[:2]
        keypoints = pose["keypoints"]
        confidences = pose["confidences"]
        box = pose["box"]
        box_width = max(1.0, float(box[2]) - float(box[0]))
        box_height = max(1.0, float(box[3]) - float(box[1]))
        box_aspect = box_width / box_height

        left_shoulder = self._point(keypoints, confidences, 5, 0.20)
        right_shoulder = self._point(keypoints, confidences, 6, 0.20)
        left_wrist = self._point(keypoints, confidences, 9, 0.20)
        right_wrist = self._point(keypoints, confidences, 10, 0.20)
        left_hip = self._point(keypoints, confidences, 11, 0.20)
        right_hip = self._point(keypoints, confidences, 12, 0.20)
        left_knee = self._point(keypoints, confidences, 13, 0.20)
        right_knee = self._point(keypoints, confidences, 14, 0.20)
        left_ankle = self._point(keypoints, confidences, 15, 0.20)
        right_ankle = self._point(keypoints, confidences, 16, 0.20)
        shoulder_mid = self._midpoint(left_shoulder, right_shoulder)
        hip_mid = self._midpoint(left_hip, right_hip)

        torso_verticality = 1.0
        if shoulder_mid is not None and hip_mid is not None:
            torso_length = math.hypot(
                hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]
            )
            if torso_length > 1e-6:
                torso_verticality = abs(hip_mid[1] - shoulder_mid[1]) / torso_length

        knee_angles = [
            self._angle(left_hip, left_knee, left_ankle),
            self._angle(right_hip, right_knee, right_ankle),
        ]
        knee_angles = [value for value in knee_angles if value is not None]
        knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else None
        posture_aspect = box_aspect
        lying_like = torso_verticality < 0.45 or (posture_aspect > 1.20 and torso_verticality < 0.70)
        upright_like = torso_verticality > 0.65 and posture_aspect < 1.10

        def wrist_data(wrist, shoulder):
            if wrist is None or shoulder is None:
                return False, None, None
            return (
                wrist[1] < shoulder[1] + box_height * 0.03,
                wrist[0] / max(1.0, float(width)),
                wrist[1] / max(1.0, float(height)),
            )

        left_above, left_wrist_x, left_wrist_y = wrist_data(left_wrist, left_shoulder)
        right_above, right_wrist_x, right_wrist_y = wrist_data(right_wrist, right_shoulder)

        bent_knee = knee_angle is not None and knee_angle < 150.0
        knees_near_hips = False
        if hip_mid is not None:
            knee_mid = self._midpoint(left_knee, right_knee)
            if knee_mid is not None:
                knees_near_hips = abs(knee_mid[1] - hip_mid[1]) < box_height * 0.45
        sitting_like = (
            not lying_like
            and torso_verticality > 0.42
            and (bent_knee or knees_near_hips)
        )
        body_points = [
            point
            for point in (
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip,
                left_knee,
                right_knee,
            )
            if point is not None
        ]
        if body_points:
            body_x = [point[0] for point in body_points]
            body_y = [point[1] for point in body_points]
            body_pad_x = box_width * 0.08
            body_pad_y = box_height * 0.08
            body_bbox = (
                max(0.0, min(body_x) - body_pad_x),
                max(0.0, min(body_y) - body_pad_y),
                min(float(width - 1), max(body_x) + body_pad_x),
                min(float(height - 1), max(body_y) + body_pad_y),
            )
        else:
            body_bbox = tuple(float(value) for value in box)
        return {
            "bbox": tuple(float(value) for value in box),
            "body_bbox": body_bbox,
            "body_anchors": body_points,
            "center_y": 0.5 * (float(box[1]) + float(box[3])) / max(1.0, float(height)),
            "height": box_height / max(1.0, float(height)),
            "aspect": posture_aspect,
            "torso_verticality": torso_verticality,
            "lying_like": lying_like,
            "upright_like": upright_like,
            "sitting_like": sitting_like,
            "left_wrist_above": left_above,
            "right_wrist_above": right_above,
            "left_wrist_x": left_wrist_x,
            "right_wrist_x": right_wrist_x,
            "left_wrist_y": left_wrist_y,
            "right_wrist_y": right_wrist_y,
        }

    def _predict(self, frame):
        """跑一次姿态推理。

        GPU 半精度：新版本 ultralytics 把 half 改名成 quantize(16=FP16)，
        旧版本仍叫 half。先试新名，失败再退回旧名，避免每帧刷弃用警告。
        """
        kwargs = dict(
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )
        if self.half and str(self.device).startswith("cuda"):
            try:
                return self.model.predict(frame, quantize=16, **kwargs)
            except Exception:
                return self.model.predict(frame, half=True, **kwargs)
        return self.model.predict(frame, **kwargs)

    def extract_features(self, frames, target_center=None, target_centers=None):
        if not self.ready or self.model is None:
            return []
        features = []
        for frame_index, frame in enumerate(frames):
            try:
                results = self._predict(frame)
            except Exception:
                continue
            if not results:
                continue
            frame_target_center = target_center
            if target_centers is not None and frame_index < len(target_centers):
                frame_target_center = target_centers[frame_index]
            pose = self._select_person(
                results[0],
                frame.shape[1],
                target_center=frame_target_center,
            )
            if pose is not None:
                feature = self._feature(pose, frame.shape)
                feature["frame_index"] = frame_index
                features.append(feature)
        return features

    def warmup(self, frame):
        if not self.ready or frame is None:
            return
        self.extract_features([frame])

    @staticmethod
    def _motion(values):
        values = [value for value in values if value is not None]
        if len(values) < 3:
            return 0.0, 0.0, 0
        changes = [values[index] - values[index - 1] for index in range(1, len(values))]
        direction_changes = sum(
            1 for index in range(1, len(changes))
            if changes[index] * changes[index - 1] < 0.0
        )
        return max(values) - min(values), sum(abs(value) for value in changes), direction_changes

    def classify(self, features):
        if len(features) < 3:
            return ACTION_UNKNOWN, 0.0, "too few pose frames"

        split = max(2, int(math.ceil(len(features) / 4.0)))
        first = features[:split]
        last = features[-split:]
        median = lambda values: sorted(values)[len(values) // 2]
        first_center = median([value["center_y"] for value in first])
        last_center = median([value["center_y"] for value in last])
        first_torso = median([value["torso_verticality"] for value in first])
        last_torso = median([value["torso_verticality"] for value in last])
        first_aspect = median([value["aspect"] for value in first])
        last_aspect = median([value["aspect"] for value in last])
        first_height = median([value["height"] for value in first])
        last_height = median([value["height"] for value in last])
        first_lie_ratio = sum(value["lying_like"] for value in first) / float(len(first))
        last_lie_ratio = sum(value["lying_like"] for value in last) / float(len(last))
        first_upright_ratio = sum(value["upright_like"] for value in first) / float(len(first))
        last_sitting_ratio = sum(value["sitting_like"] for value in last) / float(len(last))

        center_values = [value["center_y"] for value in features]
        torso_values = [value["torso_verticality"] for value in features]
        aspect_values = [value["aspect"] for value in features]
        height_values = [value["height"] for value in features]
        max_center_drop = max(
            (
                center_values[index] - center_values[index - 1]
                for index in range(1, len(center_values))
            ),
            default=0.0,
        )
        max_two_frame_center_drop = max(
            (
                center_values[index] - center_values[index - 2]
                for index in range(2, len(center_values))
            ),
            default=0.0,
        )
        minimum_torso = min(torso_values)
        maximum_aspect = max(aspect_values)
        minimum_height = min(height_values)
        rapid_torso_drop = first_torso - minimum_torso
        rapid_aspect_gain = maximum_aspect - first_aspect
        rapid_height_loss = (
            1.0 - minimum_height / first_height
            if first_height > 1e-6
            else 0.0
        )
        final_posture_is_low = bool(
            last_lie_ratio >= 0.30 or last_sitting_ratio >= 0.40
        )

        center_drop = last_center - first_center
        torso_drop = first_torso - last_torso
        aspect_gain = last_aspect - first_aspect
        height_ratio = last_height / first_height if first_height > 1e-6 else 1.0
        lie_gain = last_lie_ratio - first_lie_ratio
        transition_signals = sum(
            signal for signal in (
                center_drop > 0.035,
                torso_drop > 0.15,
                aspect_gain > 0.18,
                height_ratio < 0.88,
                lie_gain > 0.25,
            )
        )
        rapid_fall_signals = sum(
            signal for signal in (
                max_center_drop > 0.035,
                max_two_frame_center_drop > 0.055,
                rapid_torso_drop > 0.18,
                rapid_aspect_gain > 0.18,
                rapid_height_loss > 0.10,
            )
        )
        impact_signal = bool(
            max_two_frame_center_drop > 0.065
            or rapid_torso_drop > 0.20
            or rapid_aspect_gain > 0.22
            or (
                rapid_height_loss > 0.15
                and rapid_torso_drop > 0.15
            )
        )
        clear_fall = (
            first_upright_ratio >= 0.25
            and first_lie_ratio <= 0.50
            and final_posture_is_low
            and rapid_fall_signals >= 3
            and impact_signal
        )
        relaxed_fall = (
            first_upright_ratio >= 0.30
            and first_lie_ratio <= 0.50
            and final_posture_is_low
            and (
                transition_signals >= 3
                or rapid_fall_signals >= 4
            )
            and impact_signal
        )
        if clear_fall or relaxed_fall:
            confidence = min(
                0.98,
                0.60
                + 0.06 * max(transition_signals, rapid_fall_signals)
                + 0.08 * bool(max_two_frame_center_drop > 0.055),
            )
            return "sudden_fall", confidence, "pose fall transition"

        wave_scores = []
        for side in ("left", "right"):
            above_key = "%s_wrist_above" % side
            x_key = "%s_wrist_x" % side
            y_key = "%s_wrist_y" % side
            raised_points = [
                (value[x_key], value[y_key])
                for value in features
                if value[above_key]
                and value[x_key] is not None
                and value[y_key] is not None
            ]
            if len(raised_points) < 3:
                continue
            raised_x = [point[0] for point in raised_points]
            raised_y = [point[1] for point in raised_points]
            x_range, x_path, x_direction_changes = self._motion(raised_x)
            y_range, y_path, y_direction_changes = self._motion(raised_y)
            raised_ratio = len(raised_points) / float(len(features))
            motion_range = max(x_range, y_range)
            total_path = x_path + y_path
            direction_changes = x_direction_changes + y_direction_changes
            has_reversal = direction_changes >= 1 or total_path >= motion_range * 1.35
            if (
                raised_ratio >= 0.30
                and motion_range >= 0.035
                and total_path >= 0.060
                and has_reversal
            ):
                score = min(
                    0.98,
                    0.55
                    + motion_range * 2.2
                    + min(0.20, direction_changes * 0.05),
                )
                wave_scores.append(score)
        if wave_scores:
            return "waving", max(wave_scores), "raised wrist motion"

        if last_lie_ratio >= 0.40:
            return (
                "lying",
                min(0.95, 0.55 + 0.35 * last_lie_ratio),
                "horizontal pose",
            )
        if last_sitting_ratio >= 0.40:
            return (
                "sitting",
                min(0.90, 0.50 + 0.35 * last_sitting_ratio),
                "bent-knee seated pose",
            )
        return ACTION_UNKNOWN, 0.0, "no supported pose action"


class PointCloudGroundAnalyzer:
    def __init__(
        self,
        camera_height,
        ground_height_limit,
        furniture_height_limit,
        max_age,
        stride,
        min_samples,
        roi_padding,
        depth_percentile,
        surface_band,
        frame_mode,
        anchor_radius,
        local_ground_padding,
        local_ground_elevation_delta,
        elevated_context_min_height=0.15,
    ):
        self.camera_height = camera_height
        # 地面线拟合结果 y = slope*z + intercept（光学坐标系）。
        # 相机俯视时地面在图像里不是水平线，必须用这条线算"离地高度"，
        # 不能用固定的 camera_height - y（那个公式只对水平相机成立）。
        self.floor_slope = None
        self.floor_intercept = None
        self.ground_height_limit = ground_height_limit
        self.furniture_height_limit = furniture_height_limit
        self.max_age = max_age
        self.stride = max(1, stride)
        self.min_samples = max(8, min_samples)
        self.roi_padding = max(0.0, min(0.45, roi_padding))
        self.depth_percentile = max(5.0, min(50.0, depth_percentile))
        self.surface_band = max(0.05, surface_band)
        self.frame_mode = frame_mode if frame_mode in ("auto", "optical", "base") else "auto"
        self.anchor_radius = max(4.0, float(anchor_radius))
        self.local_ground_padding = max(0.25, min(1.50, float(local_ground_padding)))
        self.local_ground_elevation_delta = max(0.12, float(local_ground_elevation_delta))
        # "高于本地地面"这条兜底规则的附加前提：周围那一圈本身也得离地。
        # 否则"高于地面 0.24m"量的可能只是**人的身体厚度**——
        # 实测：人平躺在地上时，胸口离地约 0.27m，人体框周围的 15% 分位是
        # 地面(-0.02m)，差值 0.29m 就超过了 0.24 的阈值，于是"躺在地上"
        # 被判成"躺在家具上"(lying)，而不是摔倒(fallen)。
        self.elevated_context_min_height = max(
            0.0, float(elevated_context_min_height)
        )

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        position = (len(ordered) - 1) * percentile / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def _mode_for_cloud(self, cloud):
        frame_id = str(getattr(getattr(cloud, "header", None), "frame_id", "")).lower()
        if self.frame_mode == "optical":
            return "optical"
        if self.frame_mode == "base":
            return "base"
        if "optical" in frame_id:
            return "optical"
        return "base"

    @staticmethod
    def _clamp_int(value, lower, upper):
        return max(lower, min(upper, int(value)))

    def _read_points(self, cloud, uvs):
        points = []
        try:
            for x, y, z in pc2.read_points(
                cloud,
                field_names=("x", "y", "z"),
                skip_nans=True,
                uvs=uvs,
            ):
                values = (float(x), float(y), float(z))
                if all(math.isfinite(value) for value in values):
                    points.append(values)
        except Exception as exc:
            return points, "point cloud read failed: %s" % exc
        return points, ""

    def optical_height(self, point):
        """光学坐标系点 -> 离地高度（米）。

        相机俯视时，地面在图像里不是一条水平线：同一个地面越远，y 越小。
        所以不能用固定的 camera_height - y（那个只对水平相机成立），
        而要用标定出来的地面线 y_floor(z) = a*z + b，取点到这条线的垂距。
        """
        _x, y, z = point
        if self.floor_slope is not None and self.floor_intercept is not None:
            floor_y = self.floor_slope * z + self.floor_intercept
            return (floor_y - y) / math.sqrt(1.0 + self.floor_slope * self.floor_slope)
        return self.camera_height - y

    def estimate_floor_line(
        self,
        cloud,
        band_count=6,
        lower_ratio=0.55,
        upper_ratio=0.98,
        min_samples=15,
    ):
        """从点云拟合地面线 y = a*z + b（光学坐标系）。

        相机俯视时，图像里不同高度的行对应不同的地面距离。把画面下半部分
        按行分成若干带，每带取"最靠下（y 最大）的那批点"作为地面，得到
        (z, y) 样本，再最小二乘拟合。返回 (a, b) 或 None。
        """
        if cloud is None or self._mode_for_cloud(cloud) != "optical":
            return None
        cloud_height = int(getattr(cloud, "height", 0))
        cloud_width = int(getattr(cloud, "width", 0))
        if cloud_height <= 1 or cloud_width <= 1:
            return None

        v_start = self._clamp_int(cloud_height * lower_ratio, 0, cloud_height - 1)
        v_end = max(v_start + 1, self._clamp_int(cloud_height * upper_ratio, 0, cloud_height))
        u1 = self._clamp_int(cloud_width * 0.15, 0, cloud_width - 1)
        u2 = max(u1 + 1, self._clamp_int(cloud_width * 0.85, 0, cloud_width))

        samples = []
        for index in range(band_count):
            v1 = v_start + int(round((v_end - v_start) * index / float(band_count)))
            v2 = max(v1 + 1, v_start + int(round((v_end - v_start) * (index + 1) / float(band_count))))
            uvs = [(u, v) for v in range(v1, v2, 2) for u in range(u1, u2, 4)]
            points, _ = self._read_points(cloud, uvs)
            usable = [
                point
                for point in points
                if math.isfinite(point[1]) and 0.3 <= point[2] <= 6.0
            ]
            if len(usable) < min_samples:
                continue
            y_floor = self._percentile([point[1] for point in usable], 85.0)
            near_floor = [
                point[2] for point in usable if point[1] >= y_floor - 0.03
            ]
            if len(near_floor) < 3:
                continue
            samples.append((self._percentile(near_floor, 50.0), y_floor))

        if len(samples) < 2:
            return None

        count = float(len(samples))
        sum_z = sum(sample[0] for sample in samples)
        sum_y = sum(sample[1] for sample in samples)
        sum_zz = sum(sample[0] * sample[0] for sample in samples)
        sum_zy = sum(sample[0] * sample[1] for sample in samples)
        denom = count * sum_zz - sum_z * sum_z
        if abs(denom) < 1e-9:
            return None
        slope = (count * sum_zy - sum_z * sum_y) / denom
        intercept = (sum_y - slope * sum_z) / count
        if not (math.isfinite(slope) and math.isfinite(intercept)):
            return None
        return slope, intercept

    def estimate_camera_height(
        self,
        cloud,
        lower_ratio=0.80,
        upper_ratio=0.98,
        side_margin=0.25,
        min_depth=0.4,
        max_depth=3.0,
        min_samples=40,
    ):
        """从"画面最下方那一带地面"估计相机离地高度（米）。

        原理：Kinect 大致水平安装时，地面在光学坐标系里恒为 y ≈ 相机高度
        （光学系 y 轴朝下）。地面上的物体只会让 y 变小、不会变大，所以取
        画面底部那一带的 y 中位数，就得到相机离地高度。

        只对光学坐标系（kinect2_rgb_optical_frame）有效；base 坐标系本来
        就以地面为原点，不需要标定。
        """
        if cloud is None:
            return None
        if self._mode_for_cloud(cloud) != "optical":
            return None
        cloud_height = int(getattr(cloud, "height", 0))
        cloud_width = int(getattr(cloud, "width", 0))
        if cloud_height <= 1 or cloud_width <= 1:
            return None

        v1 = self._clamp_int(cloud_height * lower_ratio, 0, cloud_height - 1)
        v2 = max(v1 + 1, self._clamp_int(cloud_height * upper_ratio, 0, cloud_height))
        u1 = self._clamp_int(cloud_width * side_margin, 0, cloud_width - 1)
        u2 = max(u1 + 1, self._clamp_int(cloud_width * (1.0 - side_margin), 0, cloud_width))

        uvs = [(u, v) for v in range(v1, v2, 2) for u in range(u1, u2, 2)]
        points, _ = self._read_points(cloud, uvs)
        if not points:
            return None
        # 取深度合适的点：太近可能拍到机身，太远误差大
        ys = [
            point[1]
            for point in points
            if min_depth <= point[2] <= max_depth
        ]
        if len(ys) < min_samples:
            return None
        return self._percentile(ys, 50.0)

    def analyze(self, cloud, cloud_age, image_shape, bbox, anchors=None):
        if cloud is None:
            return {"status": "unknown", "reason": "no point cloud"}
        if cloud_age is not None and cloud_age > self.max_age:
            return {
                "status": "unknown",
                "reason": "point cloud stale: %.2fs" % cloud_age,
            }
        if int(getattr(cloud, "height", 0)) <= 1 or int(getattr(cloud, "width", 0)) <= 1:
            return {"status": "unknown", "reason": "point cloud is not organized"}
        if bbox is None or len(bbox) != 4:
            return {"status": "unknown", "reason": "no pose bounding box"}

        image_height, image_width = image_shape[:2]
        cloud_width = int(cloud.width)
        cloud_height = int(cloud.height)
        scale_x = float(cloud_width) / max(1.0, float(image_width))
        scale_y = float(cloud_height) / max(1.0, float(image_height))

        raw_x1, raw_y1, raw_x2, raw_y2 = [float(value) for value in bbox]
        box_width = max(2.0, raw_x2 - raw_x1)
        box_height = max(2.0, raw_y2 - raw_y1)
        x1 = raw_x1 - box_width * self.roi_padding
        x2 = raw_x2 + box_width * self.roi_padding
        y1 = raw_y1 - box_height * self.roi_padding
        y2 = raw_y2 + box_height * self.roi_padding
        roi_x1 = self._clamp_int(x1 * scale_x, 0, cloud_width - 1)
        roi_x2 = max(roi_x1 + 1, min(cloud_width, int(x2 * scale_x)))
        roi_y1 = self._clamp_int(y1 * scale_y, 0, cloud_height - 1)
        roi_y2 = max(roi_y1 + 1, min(cloud_height, int(y2 * scale_y)))

        body_x1 = self._clamp_int(raw_x1 * scale_x, 0, cloud_width - 1)
        body_x2 = max(body_x1 + 1, min(cloud_width, int(raw_x2 * scale_x)))
        body_y1 = self._clamp_int(raw_y1 * scale_y, 0, cloud_height - 1)
        body_y2 = max(body_y1 + 1, min(cloud_height, int(raw_y2 * scale_y)))

        context_pad = self.local_ground_padding
        context_x1 = self._clamp_int((raw_x1 - box_width * context_pad) * scale_x, 0, cloud_width - 1)
        context_x2 = max(
            context_x1 + 1,
            min(cloud_width, int((raw_x2 + box_width * context_pad) * scale_x)),
        )
        context_y1 = self._clamp_int((raw_y1 - box_height * 0.35) * scale_y, 0, cloud_height - 1)
        context_y2 = max(
            context_y1 + 1,
            min(cloud_height, int((raw_y2 + box_height * context_pad) * scale_y)),
        )

        bbox_uvs = [
            (u, v)
            for v in range(roi_y1, roi_y2, self.stride)
            for u in range(roi_x1, roi_x2, self.stride)
        ]

        context_uvs = []
        for v in range(context_y1, context_y2, self.stride):
            for u in range(context_x1, context_x2, self.stride):
                if body_x1 <= u <= body_x2 and body_y1 <= v <= body_y2:
                    continue
                context_uvs.append((u, v))

        anchor_uvs = []
        if anchors:
            seen = set()
            for anchor_x, anchor_y in anchors:
                ax1 = self._clamp_int((float(anchor_x) - self.anchor_radius) * scale_x, 0, cloud_width - 1)
                ax2 = max(ax1 + 1, min(cloud_width, int((float(anchor_x) + self.anchor_radius) * scale_x)))
                ay1 = self._clamp_int((float(anchor_y) - self.anchor_radius) * scale_y, 0, cloud_height - 1)
                ay2 = max(ay1 + 1, min(cloud_height, int((float(anchor_y) + self.anchor_radius) * scale_y)))
                for v in range(ay1, ay2, self.stride):
                    for u in range(ax1, ax2, self.stride):
                        key = (u, v)
                        if key not in seen:
                            seen.add(key)
                            anchor_uvs.append(key)

        points, read_error = self._read_points(cloud, anchor_uvs) if anchor_uvs else ([], "")
        sample_source = "anchors" if len(points) >= self.min_samples else "body_bbox"
        if len(points) < self.min_samples:
            points, read_error = self._read_points(cloud, bbox_uvs)
        if read_error and not points:
            return {"status": "unknown", "reason": read_error}

        if len(points) < self.min_samples:
            return {
                "status": "unknown",
                "reason": "too few point cloud samples: %d" % len(points),
            }

        mode = self._mode_for_cloud(cloud)
        if mode == "optical":
            transformed = [
                (point[2], -point[0], self.optical_height(point))
                for point in points
            ]
        else:
            transformed = [(point[0], point[1], point[2]) for point in points]

        valid = [
            value
            for value in transformed
            if 0.25 <= value[0] <= 8.0
            and abs(value[1]) <= 8.0
            and -0.20 <= value[2] <= 2.50
        ]
        if len(valid) < self.min_samples:
            return {
                "status": "unknown",
                "reason": "too few valid point cloud samples: %d" % len(valid),
            }

        context_points = []
        local_floor_height = None
        local_floor_samples = 0
        if context_uvs:
            raw_context_points, _ = self._read_points(cloud, context_uvs)
            if mode == "optical":
                transformed_context = [
                    (point[2], -point[0], self.optical_height(point))
                    for point in raw_context_points
                ]
            else:
                transformed_context = [
                    (point[0], point[1], point[2]) for point in raw_context_points
                ]
            context_points = [
                value
                for value in transformed_context
                if 0.25 <= value[0] <= 8.0
                and abs(value[1]) <= 8.0
                and -0.20 <= value[2] <= 2.50
            ]

        nearest_depth = self._percentile([value[0] for value in valid], self.depth_percentile)
        if context_points and nearest_depth is not None:
            local_depth_window = max(0.75, self.surface_band * 3.0)
            local_context = [
                value for value in context_points
                if value[0] <= nearest_depth + local_depth_window
            ]
            if len(local_context) >= self.min_samples:
                local_floor_samples = len(local_context)
                local_floor_height = self._percentile([value[2] for value in local_context], 15.0)
        surface = [
            value for value in valid
            if nearest_depth is not None and value[0] <= nearest_depth + self.surface_band
        ]
        if len(surface) < self.min_samples:
            surface = valid

        heights = [value[2] for value in surface]
        p20 = self._percentile(heights, 20.0)
        median = self._percentile(heights, 50.0)
        p80 = self._percentile(heights, 80.0)
        ground_fraction = sum(
            1 for height in heights if height <= self.ground_height_limit
        ) / float(len(heights))
        elevated_fraction = sum(
            1 for height in heights if height >= self.furniture_height_limit
        ) / float(len(heights))

        strong_ground = bool(
            median is not None
            and median <= self.ground_height_limit
            and ground_fraction >= 0.50
        )
        low_band_ground = bool(
            p20 is not None
            and p80 is not None
            and p20 <= self.ground_height_limit
            and p80 <= self.furniture_height_limit
            and ground_fraction >= 0.45
        )
        elevated_surface = bool(
            median is not None
            and (
                median >= self.furniture_height_limit
                or elevated_fraction >= 0.45
            )
        )

        height_above_local_floor = None
        low_height_above_local_floor = None
        elevated_by_context = False
        if local_floor_height is not None:
            if median is not None:
                height_above_local_floor = median - local_floor_height
            if p20 is not None:
                low_height_above_local_floor = p20 - local_floor_height
            elevated_by_context = bool(
                height_above_local_floor is not None
                and low_height_above_local_floor is not None
                and height_above_local_floor >= self.local_ground_elevation_delta
                and low_height_above_local_floor >= self.local_ground_elevation_delta * 0.55
                # 关键前提：周围那一圈本身也得离地。
                # 躺在地上时周围一圈就是地面(≈0)，这里的差值 0.29m 只是
                # "人体厚度"，不能当成"躺在家具上"。
                and local_floor_height is not None
                and local_floor_height >= self.elevated_context_min_height
            )

        if elevated_by_context:
            status = "elevated"
        elif strong_ground or low_band_ground:
            status = "ground"
        elif elevated_surface:
            status = "elevated"
        else:
            status = "unknown"

        ground_confident = bool(status == "ground")
        if ground_confident and low_height_above_local_floor is not None:
            # 用"身体最贴近地面的那部分(p20)"判断，而不是人体采样中位数：
            # 人躺在地上时身体本身就有 20~30cm 厚，中位数会被自己的厚度顶出阈值，
            # 结果"躺在地上"反而被判成"不是地面"。
            ground_confident = (
                low_height_above_local_floor
                <= self.local_ground_elevation_delta * 0.90
            )
        elif ground_confident and height_above_local_floor is not None:
            ground_confident = height_above_local_floor <= self.local_ground_elevation_delta * 0.90
        elif ground_confident and median is not None:
            ground_confident = bool(
                median <= self.ground_height_limit * 0.75
                or ground_fraction >= 0.65
            )

        frame_id = str(getattr(getattr(cloud, "header", None), "frame_id", ""))
        return {
            "status": status,
            "ground_like": status == "ground",
            "ground_confident": ground_confident,
            "ground_height_limit": self.ground_height_limit,
            "furniture_height_limit": self.furniture_height_limit,
            "surface_height_median": median,
            "surface_height_p20": p20,
            "surface_height_p80": p80,
            "local_floor_height_p15": local_floor_height,
            "height_above_local_floor": height_above_local_floor,
            "low_height_above_local_floor": low_height_above_local_floor,
            "elevated_by_context": elevated_by_context,
            "ground_fraction": ground_fraction,
            "elevated_fraction": elevated_fraction,
            "samples": len(surface),
            "local_floor_samples": local_floor_samples,
            "sample_source": sample_source,
            "mode": mode,
            "frame": frame_id,
            "reason": "surface median=%.3fm p20=%.3fm local_floor=%.3fm delta=%.3fm status=%s"
            % (
                median if median is not None else -1.0,
                p20 if p20 is not None else -1.0,
                local_floor_height if local_floor_height is not None else -1.0,
                height_above_local_floor if height_above_local_floor is not None else -1.0,
                status,
            ),
        }


class PoseActionRecognitionNode:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", DEFAULT_IMAGE_TOPIC)
        self.points_topic = rospy.get_param("~points_topic", DEFAULT_POINTS_TOPIC)
        self.say_topic = rospy.get_param("~say_topic", DEFAULT_SAY_TOPIC)
        self.result_topic = rospy.get_param("~result_topic", "~result")
        self.camera_wait_timeout = float(
            rospy.get_param("~camera_wait_timeout", rospy.get_param("~llm_timeout", 90.0))
        )
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.window_name = rospy.get_param("~window_name", "主人动作识别")
        self.auto_analyze = bool(rospy.get_param("~auto_analyze", True))
        self.auto_repeat_seconds = float(rospy.get_param("~auto_repeat_seconds", 0.0))
        self.startup_speech = rospy.get_param("~startup_speech", "正在识别，请稍候。")
        self.startup_settle_seconds = max(
            0.0,
            float(rospy.get_param("~startup_settle_seconds", 0.0)),
        )
        self.capture_delay = float(rospy.get_param("~capture_delay", 0.3))
        self.capture_duration = max(0.0, float(rospy.get_param("~capture_duration", 5.0)))
        self.capture_frame_count = max(1, int(rospy.get_param("~capture_frame_count", 9)))
        self.pose_enabled = bool(rospy.get_param("~pose_enabled", True))
        configured_pose_model = rospy.get_param("~pose_model_path", "")
        pose_model_path = PoseActionAnalyzer.resolve_model_path(configured_pose_model)
        self.pose_analyzer = PoseActionAnalyzer(
            pose_model_path,
            rospy.get_param("~pose_device", "cpu"),
            max(160, int(rospy.get_param("~pose_image_size", 416))),
            float(rospy.get_param("~pose_confidence", 0.25)),
            float(rospy.get_param("~pose_iou", 0.45)),
            max(1, int(rospy.get_param("~pose_max_detections", 4))),
            bool(rospy.get_param("~pose_half", False)),
        )
        self.pointcloud_enabled = bool(rospy.get_param("~pointcloud_enabled", True))
        self.pointcloud_analyzer = PointCloudGroundAnalyzer(
            float(rospy.get_param("~pointcloud_camera_height", 0.85)),
            float(rospy.get_param("~pointcloud_ground_height_limit", 0.35)),
            float(rospy.get_param("~pointcloud_furniture_height_limit", 0.40)),
            max(0.1, float(rospy.get_param("~pointcloud_max_age", 1.0))),
            max(1, int(rospy.get_param("~pointcloud_stride", 8))),
            max(8, int(rospy.get_param("~pointcloud_min_samples", 30))),
            float(rospy.get_param("~pointcloud_roi_padding", 0.20)),
            float(rospy.get_param("~pointcloud_depth_percentile", 20.0)),
            float(rospy.get_param("~pointcloud_surface_band", 0.25)),
            rospy.get_param("~pointcloud_frame_mode", "auto"),
            float(rospy.get_param("~pointcloud_anchor_radius", 14.0)),
            float(rospy.get_param("~pointcloud_local_ground_padding", 0.85)),
            float(rospy.get_param("~pointcloud_elevated_delta", 0.24)),
            float(rospy.get_param("~pointcloud_elevated_context_min_height", 0.15)),
        )
        # 相机离地高度自动标定：拆装机器人后相机高度会变，
        # 开机时用点云现场测一次，比写死 0.85 可靠。
        self.auto_camera_height = bool(rospy.get_param("~auto_camera_height", True))
        self.camera_height_calibration_timeout = max(
            1.0, float(rospy.get_param("~camera_height_calibration_timeout", 8.0))
        )
        self.camera_height_calibration_samples = max(
            1, int(rospy.get_param("~camera_height_calibration_samples", 5))
        )
        self.camera_height_min = float(rospy.get_param("~camera_height_min", 0.30))
        self.camera_height_max = float(rospy.get_param("~camera_height_max", 1.80))
        self.say_wait_timeout = float(rospy.get_param("~say_wait_timeout", 5.0))
        self.model_warmup_retries = max(1, int(rospy.get_param("~model_warmup_retries", 3)))
        self.model_warmup_retry_delay = max(
            0.0, float(rospy.get_param("~model_warmup_retry_delay", 2.0))
        )
        self.use_owner_roi = bool(rospy.get_param("~use_owner_roi", False))
        self.owner_roi = parse_owner_roi(rospy.get_param("~owner_roi", ""))
        self.roi_padding = max(
            0.0,
            min(1.0, float(rospy.get_param("~roi_padding", 0.55))),
        )
        self.overlay_hold_seconds = max(
            0.0, float(rospy.get_param("~overlay_hold_seconds", 6.0))
        )
        self.overlay_lock = threading.Lock()
        self.overlay_frame = None
        self.overlay_until = 0.0
        # 外部触发（供任务节点调用）：每收到一次触发做一次识别，
        # 可携带该次请求的 ROI 与 request_id，结果里原样带回 request_id。
        self.trigger_topic = rospy.get_param("~trigger_topic", "~trigger")
        self.speak_enabled = bool(rospy.get_param("~speak_enabled", True))
        # 被外部触发时默认不播报（调用方自己决定说什么），置 true 可让它也播报
        self.speak_on_trigger = bool(rospy.get_param("~speak_on_trigger", False))
        self.request_lock = threading.Lock()
        self.pending_request_id = ""
        self.pending_roi = None
        self.active_request_id = ""
        self.active_roi = None
        self.last_action = ACTION_UNKNOWN
        self.last_place = PLACE_UNKNOWN
        self.last_pose_action = ACTION_UNKNOWN
        self.last_confidence = 0.0
        self.last_ground_status = "unknown"

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = 0.0
        self.latest_pointcloud = None
        self.latest_pointcloud_stamp = 0.0
        self.capture_full_frames = []
        self.capture_rois = []

        self.bridge = CvBridge()
        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=10)
        self.result_pub = rospy.Publisher(self.result_topic, String, queue_size=10)
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.points_sub = None
        if self.pointcloud_enabled:
            self.points_sub = rospy.Subscriber(
                self.points_topic,
                PointCloud2,
                self.pointcloud_callback,
                queue_size=1,
                buff_size=2**24,
            )
        self.trigger_sub = rospy.Subscriber(
            self.trigger_topic,
            String,
            self.trigger_callback,
            queue_size=5,
        )
        rospy.loginfo("Action recognition trigger topic: %s", self.trigger_topic)
        # "我已就绪"信号：模型加载 + 预热 + 地面线标定全部完成之后才发。
        # 调用方（复用模式）会等这条消息再发第一次触发，避免开局那几次
        # 触发被"还没预热好"挡掉、白等好几秒。latch=True 让晚连的订阅者
        # 也能立刻收到。
        self.ready_pub = rospy.Publisher("~ready", String, queue_size=1, latch=True)

        self.inference_lock = threading.Lock()
        self.last_inference_time = 0.0
        self.window_available = self.show_window
        self.startup_announced = False
        self.warmup_done = False
        self.capture_status_lock = threading.Lock()
        self.capture_status = ""
        self.startup_thread = threading.Thread(
            target=self.startup_sequence,
            name="pose_action_startup",
            daemon=True,
        )

        rospy.loginfo("Subscribing camera: %s", self.image_topic)
        if self.pointcloud_enabled:
            rospy.loginfo("Subscribing organized point cloud: %s", self.points_topic)
        rospy.loginfo("Using YOLO Pose only for action recognition")
        if self.pose_enabled:
            pose_ready, pose_message = self.pose_analyzer.initialize()
            if pose_ready:
                rospy.loginfo(
                    "Pose action helper ready: model=%s device=%s",
                    self.pose_analyzer.model_path,
                    self.pose_analyzer.device,
                )
            else:
                rospy.logwarn("Pose action helper disabled: %s", pose_message)
        self.wait_for_camera_frame()
        if self.startup_settle_seconds > 0.0:
            rospy.sleep(self.startup_settle_seconds)
        self.startup_thread.start()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Camera image conversion failed: %s", exc)
            return

        with self.frame_lock:
            self.latest_frame = frame
            self.latest_stamp = time.time()

    def pointcloud_callback(self, msg):
        with self.frame_lock:
            self.latest_pointcloud = msg
            self.latest_pointcloud_stamp = time.time()

    def wait_for_camera_frame(self):
        deadline = time.time() + max(5.0, self.camera_wait_timeout)
        rate = rospy.Rate(30)
        started = time.time()
        self._last_wait_log = 0.0
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.frame_lock:
                has_frame = self.latest_frame is not None
            if has_frame:
                rospy.loginfo(
                    "Camera frame received on %s after %.1fs",
                    self.image_topic,
                    time.time() - started,
                )
                return
            self.log_camera_wait_state(time.time() - started)
            self.show_preview()
            rate.sleep()
        raise RuntimeError(
            "Timed out waiting for camera image: %s (waited %.0fs). %s"
            % (self.image_topic, time.time() - started, self.describe_image_topics())
        )

    @staticmethod
    def describe_image_topics():
        """查询 master 上现有的图像/点云话题，用于超时报错时定位原因。"""
        try:
            _, _, topic_types = rospy.get_master().getTopicTypes()
        except Exception as exc:
            return "无法查询 ROS master（master 没起来或连不上？）: %s" % exc
        names = sorted(
            name
            for name, _ in topic_types
            if "image" in name.lower() or "points" in name.lower()
        )
        if not names:
            return "master 上没有任何图像/点云话题：机器人栈（kinect2 驱动）没有起来"
        return "master 上现有图像/点云话题: %s" % ", ".join(names)

    def log_camera_wait_state(self, waited):
        """每 5 秒提示一次等待状态和可用话题，避免 90 秒静默后只报一句超时。"""
        now = time.time()
        if now - self._last_wait_log < 5.0:
            return
        self._last_wait_log = now
        rospy.logwarn(
            "Waiting for camera image %s (%.0fs/%.0fs). %s",
            self.image_topic,
            waited,
            max(5.0, self.camera_wait_timeout),
            self.describe_image_topics(),
        )

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_latest_pointcloud(self):
        with self.frame_lock:
            return self.latest_pointcloud, self.latest_pointcloud_stamp

    def prepare_action_frame(self, frame):
        # 本次请求若带了 ROI（外部触发），优先用它；否则退回静态 owner_roi
        roi = self.active_roi if self.active_roi is not None else self.owner_roi
        if roi is None or (not self.use_owner_roi and self.active_roi is None):
            return frame, None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = roi
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        roi_width = max(1, x2 - x1)
        roi_height = max(1, y2 - y1)
        pad_x = int(round(roi_width * self.roi_padding))
        pad_y = int(round(roi_height * self.roi_padding))
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(width, x2 + pad_x)
        crop_y2 = min(height, y2 + pad_y)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return frame, None
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            return frame, None
        return crop, (crop_x1, crop_y1, crop_x2, crop_y2)

    def set_capture_status(self, text):
        with self.capture_status_lock:
            self.capture_status = text

    def get_capture_status(self):
        with self.capture_status_lock:
            return self.capture_status

    @staticmethod
    def _geometry_in_full_frame(feature, roi):
        """把 ROI 裁剪帧里的 bbox/关键点换算回整帧坐标。"""
        if not feature:
            return None, []
        bbox = feature.get("body_bbox") or feature.get("bbox")
        anchors = list(feature.get("body_anchors") or [])
        if roi is not None:
            offset_x, offset_y = float(roi[0]), float(roi[1])
            if bbox is not None and len(bbox) >= 4:
                bbox = (
                    float(bbox[0]) + offset_x,
                    float(bbox[1]) + offset_y,
                    float(bbox[2]) + offset_x,
                    float(bbox[3]) + offset_y,
                )
            anchors = [
                (float(point[0]) + offset_x, float(point[1]) + offset_y)
                for point in anchors
            ]
        return bbox, anchors

    def render_analysis_frame(
        self,
        frame,
        feature,
        roi,
        action,
        place,
        confidence,
        pose_action,
        ground_relation,
    ):
        """在抓拍的整帧上画出人体框、关键点和动作结论。"""
        annotated = frame.copy()
        ground_relation = ground_relation or {}
        ground_status = ground_relation.get("status", "unknown")
        surface_median = ground_relation.get("surface_height_median")
        furniture_limit = ground_relation.get("furniture_height_limit")
        if surface_median is not None:
            surface_text = "%.2fm(%s)" % (surface_median, ground_status)
            if furniture_limit is not None:
                surface_text += " 家具阈值%.2fm" % furniture_limit
        else:
            surface_text = ground_status
        bbox, anchors = self._geometry_in_full_frame(feature, roi)
        height, width = annotated.shape[:2]
        if bbox is not None and len(bbox) >= 4:
            x1 = max(0, min(width - 1, int(round(float(bbox[0])))))
            y1 = max(0, min(height - 1, int(round(float(bbox[1])))))
            x2 = max(0, min(width - 1, int(round(float(bbox[2])))))
            y2 = max(0, min(height - 1, int(round(float(bbox[3])))))
            box_color = (
                (0, 0, 255) if action in ("fallen", "sudden_fall") else (0, 200, 0)
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        for point in anchors:
            center = (int(round(point[0])), int(round(point[1])))
            if 0 <= center[0] < width and 0 <= center[1] < height:
                cv2.circle(annotated, center, 4, (255, 128, 0), -1)
                cv2.circle(annotated, center, 5, (0, 0, 0), 1)

        lines = [
            (
                "动作: %s (%s)  置信度 %.2f"
                % (ACTION_TEXT.get(action, action), action, confidence),
                (0, 255, 255),
            ),
            (
                "位置: %s   姿态: %s   支撑面: %s"
                % (
                    PLACE_LABEL.get(place, place) or place,
                    ACTION_TEXT.get(pose_action, pose_action),
                    surface_text,
                ),
                (0, 220, 0),
            ),
        ]
        text_y = 30
        for text, color in lines:
            text_y = draw_text_box(annotated, text, (12, text_y), color)
        return annotated

    def set_result_overlay(
        self, features, action, place, confidence, pose_action, ground_relation
    ):
        """记录最新结论，并把带标注的抓拍帧设为预览内容。"""
        self.last_action = action
        self.last_place = place
        self.last_confidence = float(confidence)
        self.last_pose_action = pose_action
        self.last_ground_status = (ground_relation or {}).get("status", "unknown")
        if not self.window_available:
            return
        frame = self.capture_full_frames[-1] if self.capture_full_frames else None
        if frame is None:
            return
        roi = self.capture_rois[-1] if self.capture_rois else None
        feature = features[-1] if features else None
        annotated = self.render_analysis_frame(
            frame, feature, roi, action, place, confidence, pose_action, ground_relation
        )
        with self.overlay_lock:
            self.overlay_frame = annotated
            self.overlay_until = time.time() + self.overlay_hold_seconds

    def clear_result_overlay(self):
        with self.overlay_lock:
            self.overlay_frame = None
            self.overlay_until = 0.0

    def show_preview(self):
        if not self.window_available:
            return
        now = time.time()
        with self.overlay_lock:
            overlay_frame = self.overlay_frame
            overlay_until = self.overlay_until
        try:
            if overlay_frame is not None and now < overlay_until:
                display_frame = overlay_frame.copy()
                draw_text_box(
                    display_frame,
                    "识别结果（保持 %.1fs）" % max(0.0, overlay_until - now),
                    (12, display_frame.shape[0] - 14),
                    (0, 255, 255),
                )
            else:
                frame = self.get_latest_frame()
                if frame is None:
                    return
                display_frame = frame.copy()
                status = self.get_capture_status()
                if status:
                    draw_text_box(display_frame, status, (16, 34), (0, 255, 255), 0.7)
                elif self.warmup_done:
                    draw_text_box(
                        display_frame,
                        "待命：按空格识别，q 退出",
                        (16, 34),
                        (0, 255, 255),
                        0.7,
                    )
                if self.warmup_done and self.last_action != ACTION_UNKNOWN:
                    draw_text_box(
                        display_frame,
                        "上次: %s / %s (%.2f)"
                        % (
                            ACTION_TEXT.get(self.last_action, self.last_action),
                            PLACE_LABEL.get(self.last_place, self.last_place)
                            or self.last_place,
                            self.last_confidence,
                        ),
                        (16, display_frame.shape[0] - 14),
                        (0, 220, 0),
                    )
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                rospy.signal_shutdown("User closed action recognition window")
            elif key == 32 and self.warmup_done:
                self.request_inference("keyboard")
        except cv2.error as exc:
            self.window_available = False
            rospy.logwarn("Camera preview disabled: %s", exc)

    def capture_frames(self):
        frame_count = self.capture_frame_count
        duration = self.capture_duration
        if frame_count == 1:
            sample_times = [0.0]
        else:
            interval = duration / float(frame_count - 1)
            sample_times = [interval * index for index in range(frame_count)]

        frames = []
        full_frames = []
        capture_rois = []
        capture_started = time.time()
        for index, sample_time in enumerate(sample_times, start=1):
            while not rospy.is_shutdown():
                elapsed = time.time() - capture_started
                remaining = sample_time - elapsed
                if remaining <= 0.0:
                    break
                self.set_capture_status("采集中 %d/%d" % (index, frame_count))
                rospy.sleep(min(0.05, remaining))

            if rospy.is_shutdown():
                break
            frame = self.get_latest_frame()
            if frame is None:
                continue
            action_frame, roi = self.prepare_action_frame(frame)
            frames.append(action_frame)
            full_frames.append(frame)
            capture_rois.append(roi)
            self.set_capture_status("已采集 %d/%d" % (index, frame_count))

        self.set_capture_status("")
        if not frames:
            raise RuntimeError("No camera frames available for action recognition")
        self.capture_full_frames = full_frames
        self.capture_rois = capture_rois
        return frames

    def analyze_ground_relation(self, frame, pose_features):
        if not self.pointcloud_enabled:
            return {"status": "unknown", "reason": "point cloud disabled"}
        if frame is None:
            return {"status": "unknown", "reason": "no final image"}
        if not pose_features:
            return {"status": "unknown", "reason": "no pose bbox"}
        analysis_frame = frame
        final_feature = dict(pose_features[-1])
        if self.capture_full_frames and self.capture_rois and self.capture_rois[-1] is not None:
            roi = self.capture_rois[-1]
            full_frame = self.capture_full_frames[-1]
            offset_x, offset_y = roi[0], roi[1]
            for key in ("bbox", "body_bbox"):
                box = final_feature.get(key)
                if box is not None and len(box) >= 4:
                    final_feature[key] = (
                        float(box[0]) + offset_x,
                        float(box[1]) + offset_y,
                        float(box[2]) + offset_x,
                        float(box[3]) + offset_y,
                    )
            anchors = final_feature.get("body_anchors") or []
            final_feature["body_anchors"] = [
                (float(point[0]) + offset_x, float(point[1]) + offset_y)
                for point in anchors
            ]
            analysis_frame = full_frame
        bbox = final_feature.get("body_bbox") or final_feature.get("bbox")
        anchors = final_feature.get("body_anchors") or []
        cloud, cloud_stamp = self.get_latest_pointcloud()
        cloud_age = time.time() - cloud_stamp if cloud_stamp > 0.0 else None
        return self.pointcloud_analyzer.analyze(
            cloud,
            cloud_age,
            analysis_frame.shape,
            bbox,
            anchors=anchors,
        )

    def warmup_model(self):
        frame = self.get_latest_frame()
        if frame is None:
            raise RuntimeError("Cannot warm up action recognition without a camera frame")
        if not self.pose_enabled or not self.pose_analyzer.ready:
            raise RuntimeError("YOLO Pose is unavailable")
        rospy.loginfo("Warming up local pose action helper")
        self.pose_analyzer.warmup(frame)
        rospy.loginfo("Local pose action helper warm-up complete")
        self.warmup_done = True

    def calibrate_camera_height(self):
        """开机时用点云现场标定相机离地高度。

        拆装机器人后相机高度会变，写死的 0.85 会让"家具/地面"判断整体偏移，
        所以每次启动都实测一次：取画面最下方那一带的地面点，用它们的 y
        （光学系朝下）中位数作为相机离地高度。
        """
        analyzer = self.pointcloud_analyzer
        if not self.pointcloud_enabled or analyzer is None:
            return False
        if not self.auto_camera_height:
            rospy.loginfo(
                "Camera height auto-calibration disabled; using %.3f m",
                analyzer.camera_height,
            )
            return False

        deadline = time.time() + self.camera_height_calibration_timeout
        samples = []
        floor_lines = []
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and time.time() < deadline:
            cloud, cloud_stamp = self.get_latest_pointcloud()
            cloud_age = time.time() - cloud_stamp if cloud_stamp else None
            if cloud is not None and (cloud_age is None or cloud_age <= 1.0):
                value = analyzer.estimate_camera_height(cloud)
                if value is not None:
                    samples.append(value)
                line = analyzer.estimate_floor_line(cloud)
                if line is not None:
                    floor_lines.append(line)
                if (
                    len(samples) >= self.camera_height_calibration_samples
                    and len(floor_lines) >= 3
                ):
                    break
            rate.sleep()

        if floor_lines:
            # 拟合出地面线后，高度改用"点到地面线的垂距"，
            # 不再依赖相机高度/俯角，距离远近都不影响。
            slopes = sorted(value[0] for value in floor_lines)
            intercepts = sorted(value[1] for value in floor_lines)
            slope = slopes[len(slopes) // 2]
            intercept = intercepts[len(intercepts) // 2]
            analyzer.floor_slope = slope
            analyzer.floor_intercept = intercept
            rospy.loginfo(
                "Floor line calibrated: y = %.4f*z + %.4f "
                "(implied camera pitch %.1f deg, samples=%d)",
                slope,
                intercept,
                math.degrees(math.atan(-slope)),
                len(floor_lines),
            )

        if not samples:
            rospy.logwarn(
                "Camera height auto-calibration failed (no usable floor points); "
                "keeping %.3f m",
                analyzer.camera_height,
            )
            return False

        samples.sort()
        value = samples[len(samples) // 2]
        clamped = max(self.camera_height_min, min(self.camera_height_max, value))
        if abs(clamped - value) > 1e-6:
            rospy.logwarn(
                "Measured camera height %.3f m out of [%.2f, %.2f]; clamped to %.3f m",
                value,
                self.camera_height_min,
                self.camera_height_max,
                clamped,
            )
        analyzer.camera_height = clamped
        rospy.loginfo(
            "Camera height auto-calibrated: %.3f m (samples=%d, raw=%.3f..%.3f)",
            clamped,
            len(samples),
            samples[0],
            samples[-1],
        )
        return True

    def startup_sequence(self):
        try:
            self.speak(self.startup_speech)
            self.warmup_model()
            self.calibrate_camera_height()
            self.startup_announced = True
            # 一切就绪 -> 通知调用方"可以触发了"
            try:
                self.ready_pub.publish(String(data="ready"))
                rospy.loginfo("动作识别已就绪，可以触发（ready 已发布）")
            except Exception as exc:
                rospy.logwarn("发布 ready 信号失败：%s", exc)
            if self.auto_analyze:
                rospy.sleep(max(0.0, self.capture_delay))
                self.request_inference("startup")
        except Exception as exc:
            rospy.logfatal("YOLO Pose action recognition startup failed: %s", exc)
            try:
                self.speak("识别系统启动失败。")
            except Exception as speech_exc:
                rospy.logerr("Could not report startup failure by voice: %s", speech_exc)
            rospy.signal_shutdown("YOLO Pose action recognition startup failed")

    def request_inference(self, reason):
        if not self.warmup_done:
            return False
        now = time.time()
        if now - self.last_inference_time < max(0.2, self.capture_delay):
            return False
        if not self.inference_lock.acquire(False):
            rospy.loginfo_throttle(2.0, "Action recognition is still running")
            return False

        self.last_inference_time = now
        worker = threading.Thread(
            target=self._inference_worker,
            args=(reason,),
            name="pose_action_inference",
            daemon=True,
        )
        worker.start()
        return True

    def trigger_callback(self, message):
        """外部触发一次识别，供任务节点按航点调用。

        payload 支持两种写法（都通过 trigger_topic 发 std_msgs/String）：
          "x1,y1,x2,y2"                                只看这个 ROI
          {"request_id":"...","roi":"x1,y1,x2,y2"}     JSON（推荐，便于配对结果）
        留空则按全画面 / 静态 owner_roi 处理。
        """
        payload = (message.data or "").strip()
        request_id = ""
        roi = None
        if payload.startswith("{"):
            try:
                data = json.loads(payload)
            except (TypeError, ValueError) as exc:
                rospy.logwarn("Invalid action trigger payload: %s", exc)
                data = None
            if isinstance(data, dict):
                request_id = str(data.get("request_id", "") or "")
                roi = parse_owner_roi(data.get("roi"))
        else:
            roi = parse_owner_roi(payload)

        with self.request_lock:
            self.pending_request_id = request_id
            self.pending_roi = roi

        if self.request_inference("trigger"):
            rospy.loginfo(
                "Action trigger accepted: request_id=%s roi=%s", request_id or "(none)", roi
            )
            return

        # 忙或还没预热完：立刻回一条带 request_id 的错误结果，避免调用方干等
        rospy.logwarn(
            "Action trigger rejected (busy or not warm yet): request_id=%s",
            request_id or "(none)",
        )
        self.publish_trigger_error(request_id, "action recognizer busy")

    def publish_trigger_error(self, request_id, reason):
        self.result_pub.publish(
            String(
                data=json.dumps(
                    {
                        "request_id": request_id,
                        "action": ACTION_UNKNOWN,
                        "place": PLACE_UNKNOWN,
                        "recognizer": "yolo_pose",
                        "error": reason,
                    },
                    ensure_ascii=False,
                )
            )
        )

    def _inference_worker(self, reason):
        try:
            with self.request_lock:
                self.active_request_id = self.pending_request_id
                self.pending_request_id = ""
                self.active_roi = self.pending_roi
                self.pending_roi = None
            request_id = self.active_request_id
            rospy.loginfo(
                "Recognizing action (%s) request_id=%s", reason, request_id or "(none)"
            )
            self.clear_result_overlay()
            # 外部触发（任务节点调用）由调用方负责说话，这里不抢话
            if reason not in ("startup", "trigger"):
                self.speak(self.startup_speech)

            started = time.time()
            frames = self.capture_frames()
            capture_elapsed = time.time() - started
            self.set_capture_status("正在分析 %d 帧" % len(frames))
            pose_action, pose_confidence, pose_reason = ACTION_UNKNOWN, 0.0, "pose disabled"
            pose_features = []
            if self.pose_enabled and self.pose_analyzer.ready:
                try:
                    pose_features = self.pose_analyzer.extract_features(frames)
                    pose_action, pose_confidence, pose_reason = self.pose_analyzer.classify(
                        pose_features
                    )
                except Exception as exc:
                    pose_reason = "pose inference failed: %s" % exc
                    rospy.logwarn("Pose action inference failed: %s", exc)

            ground_relation = self.analyze_ground_relation(
                frames[-1] if frames else self.get_latest_frame(),
                pose_features,
            )
            total_elapsed = time.time() - started
            final_pose_is_horizontal = final_posture_is_horizontal(pose_features)
            action, place, ground_fallen, elevated_lying = merge_pose_action_result(
                pose_action,
                ground_relation,
                final_pose_is_horizontal,
            )
            speech = result_to_speech(action, place)
            self.set_result_overlay(
                pose_features,
                action,
                place,
                pose_confidence,
                pose_action,
                ground_relation,
            )
            result = {
                "request_id": request_id,
                "triggered": bool(request_id),
                "action": action,
                "place": place,
                "speech": speech,
                "recognizer": "yolo_pose",
                "raw": "",
                "qwen_action": ACTION_UNKNOWN,
                "pose_action": pose_action,
                "pose_confidence": round(pose_confidence, 3),
                "pose_reason": pose_reason,
                "pose_fall_transition": pose_action == "sudden_fall",
                "ground_relation": ground_relation,
                "ground_fallen": ground_fallen,
                "elevated_lying": elevated_lying,
                "recognition_mode": "pose_only",
                "qwen_skipped": True,
                "pose_guided_qwen": False,
                "latency_sec": 0.0,
                "capture_sec": round(capture_elapsed, 3),
                "total_sec": round(total_elapsed, 3),
                "frame_count": len(frames),
                "llm_frame_count": 0,
                "owner_roi": list(self.owner_roi) if self.owner_roi is not None else None,
                "owner_roi_applied": any(roi is not None for roi in self.capture_rois),
            }
            self.result_pub.publish(String(data=json.dumps(result, ensure_ascii=False)))
            def _fmt(value):
                return "%.2f" % value if value is not None else "n/a"

            rospy.loginfo(
                "Action: %s, place: %s, frames: %d, pose=%s(%.2f), "
                "ground=%s(median=%s p20=%s local=%s above=%s confident=%s), "
                "capture: %.2fs, total: %.2fs",
                action,
                place,
                len(frames),
                pose_action,
                pose_confidence,
                ground_relation.get("status", "unknown"),
                _fmt(ground_relation.get("surface_height_median")),
                _fmt(ground_relation.get("surface_height_p20")),
                _fmt(ground_relation.get("local_floor_height_p15")),
                _fmt(ground_relation.get("height_above_local_floor")),
                ground_relation.get("ground_confident"),
                capture_elapsed,
                total_elapsed,
            )
            if not request_id or self.speak_on_trigger:
                self.speak(speech)
        except Exception as exc:
            rospy.logerr("Action recognition failed: %s", exc)
            try:
                self.speak("识别失败，请稍后再试。")
            except Exception as speech_exc:
                rospy.logerr("Could not report recognition failure by voice: %s", speech_exc)
        finally:
            self.set_capture_status("")
            # 自动重复从“本轮结束”开始计时，保证结果画面有完整展示时间，
            # 避免识别任务首尾相接把标注画面立刻清掉。
            self.last_inference_time = time.time()
            self.inference_lock.release()

    def speak(self, text):
        if not self.speak_enabled:
            return
        text = (text or "").strip()
        if text:
            deadline = time.time() + max(0.0, self.say_wait_timeout)
            while (
                not rospy.is_shutdown()
                and self.say_pub.get_num_connections() == 0
                and time.time() < deadline
            ):
                rospy.sleep(0.05)
            if self.say_pub.get_num_connections() == 0:
                rospy.logwarn_throttle(5.0, "No TTS subscriber connected to %s", self.say_topic)
            self.say_pub.publish(String(data=text))

    def run(self):
        rate = rospy.Rate(30)
        try:
            while not rospy.is_shutdown():
                self.show_preview()
                if (
                    self.auto_repeat_seconds > 0
                    and self.startup_announced
                    and time.time() - self.last_inference_time >= self.auto_repeat_seconds
                ):
                    self.request_inference("timer")
                rate.sleep()
        finally:
            if self.window_available:
                cv2.destroyAllWindows()


def main():
    rospy.init_node("pose_action_recognition")
    try:
        PoseActionRecognitionNode().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("YOLO Pose action recognition node failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
