#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人脸 / 身形分步注册 + 识别测试。

注册流程（每位主人）
--------------------
  第 1 步  语音问姓名
  第 2 步  复述姓名并要求确认
           （话术："您说的是X，对吗？对就说确认，不对就说错误。"
             用"错了"而不是"重来"：ASR 常把"重来"听成"从来"，
             而"错误"是高频词，识别稳定得多；回答说"错了/错"也认。）
  第 3 步  只采集人脸特征（不采身形）
  第 4 步  只采集身形（ReID）特征（不重复采人脸）
  第 5 步  人脸 + 身形合并写入同一份档案

全部主人注册完后自动进入识别循环（复用现成实现）：
  判断眼前的人是不是主人、是哪一位，并播报姓名。

实现方式
--------
继承 owner_voice_reid_test.OwnerVoiceReidTest，复用现成的人脸模型、
ReID 模型、档案读写与识别循环，只覆写"注册"这一段。
本文件不修改任何已有文件。

档案位置
--------
data/reid_owner/owner_profile_<N>.npz 与 owner_profile_<N>.json
（N 由 ~owner_count 决定）

常用参数
--------
  ~owner_count              要注册几位主人（两人测试填 2）
  ~navigate_enabled         false：注册完原地进入识别循环（默认 false）
  ~enroll_retries           姓名确认被否后重问的次数（默认 3）
  ~face_capture_seconds     采人脸时长（默认沿用 record_seconds）
  ~face_sample_count        人脸目标样本数（默认 16）
  ~face_min_samples         人脸最少样本数，低于则判失败（默认 3）
  ~reuse_existing_profile   true：跳过注册直接读已有档案
"""

import os
import sys
import time
import importlib.util

import cv2
import rospy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from person_reid_owner_test import PersonReidOwnerTest  # noqa: E402
from owner_voice_reid_test import OwnerVoiceReidTest  # noqa: E402


def _find_action_core_dir():
    """定位动作识别核心（qwen_action_recognition_node.py）所在目录。

    该核心原先寄放在 offline_voice_bridge/scripts（那时它靠那个包转发
    LLM 请求）；现在它已搬进本包的 scripts 目录，和调用方同目录。
    保留"候选目录列表"的写法，方便以后整包迁移到 wpb_service_robot。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        here,
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(os.path.join(path, "qwen_action_recognition_node.py")):
            return path
    return ""


class FaceBodyEnrollTest(OwnerVoiceReidTest):
    """先采人脸、再采身形，最后合并成一份带姓名的档案。"""

    def __init__(self):
        super().__init__()
        self.enroll_retries = max(1, int(rospy.get_param("~enroll_retries", 3)))
        self.face_capture_seconds = max(
            0.5, float(rospy.get_param("~face_capture_seconds", 8.0))
        )
        self.face_sample_interval = max(
            0.05, float(rospy.get_param("~face_sample_interval", 0.30))
        )
        self.face_sample_count = max(
            1, int(rospy.get_param("~face_sample_count", 16))
        )
        self.face_min_samples = max(
            1, int(rospy.get_param("~face_min_samples", 3))
        )
        # 没有可用人脸（或人脸分数处于中间地带）时，身形分数必须达到这个
        # 更高的阈值才算主人。原逻辑这里用的是 match_threshold(0.7)，
        # 对陌生人太松（衣服颜色相近就能到 0.8+）。
        self.body_only_match_threshold = max(
            0.0, min(1.0, float(rospy.get_param("~body_only_match_threshold", 0.90)))
        )
        # 躺姿判定的补充判据（父类只认"框宽 > 高×1.25"，迎面拍到躺姿时
        # 人体框又短又不宽，宽高比到不了 1.25，会被误当成站姿）。
        self.lying_extra_height_ratio = max(
            0.05,
            min(1.0, float(rospy.get_param("~lying_extra_height_ratio", 0.70))),
        )
        self.lying_extra_aspect_min = max(
            0.05, float(rospy.get_param("~lying_extra_aspect_min", 0.55))
        )
        # "坐着但完全看不到人脸"时才走外观通道，阈值要比躺着更高：
        # 坐着的人本来应该能看到正脸，看不到说明角度很偏，风险更大。
        self.sitting_appearance_threshold = max(
            0.0,
            min(1.0, float(rospy.get_param("~sitting_appearance_threshold", 0.75))),
        )
        # "人体框明确横躺"的判定线：宽 / 高 ≥ 这个值就当横躺。
        # 用来纠正姿态判决 sitting/lying 抖动（站着的人框是竖的，坐着大概方）。
        self.strong_lying_aspect = max(
            1.0, float(rospy.get_param("~strong_lying_aspect", 1.5))
        )
        self.confirm_prompt_template = rospy.get_param(
            "~confirm_prompt_template",
            "您说的是%s，对吗？对就说确认，不对就说错误。",
        )
        # 重试时的一句话：既是"好的"的回应，也是重新提问本身。
        # 原来这里会先说"好的，请重新说出您的名字。"（约 1.6 秒），
        # 再播报一整句姓名提示（约 4 秒），一次重试要 5.6 秒；合并成一句后
        # 只要 1.8 秒左右。
        self.name_retry_prompt_text = rospy.get_param(
            "~name_retry_prompt_text", "好的，请再说一次您的名字。"
        )
        self.name_timeout_prompt_text = rospy.get_param(
            "~name_timeout_prompt_text", "我没有听清，请再说一次您的名字。"
        )
        # ASR 常把中文数字写成阿拉伯数字（"张三" -> "张3"），
        # 存进档案后播报也会念成"张3"。这里做一次规整。
        # 开销：对 2~6 个字符的字符串做 10 次替换，微秒级，可忽略。
        self.normalize_name_digits = bool(
            rospy.get_param("~normalize_name_digits", True)
        )
        self.digit_to_chinese = {
            "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
        }
        # 单帧姿态判断用的模型（只在"人脸不如"时才调用，用于确认躺/坐）
        self.pose_core = None
        self.pose_analyzer = None
        # YOLO-World 对躺姿几乎检不出来（实测：站着/坐着都有框，一躺下框
        # 就消失）。所以当它一个候选都没给时，用姿态模型在整帧上兜底找人。
        # 实测姿态模型能看到躺着的（⑩ 测试里 pose=sudden_fall(0.98)）。
        self.pose_fallback_enabled = bool(
            rospy.get_param("~pose_fallback_enabled", True)
        )
        # 姿态检测的最小间隔（秒）。姿态检测是"整帧推理"，CPU 上约 0.2~0.4 秒，
        # 所以用它来限制开销，避免每次评估都跑。
        self.pose_fallback_interval = max(
            0.0, float(rospy.get_param("~pose_fallback_interval", 0.30))
        )
        self._last_pose_fallback_time = 0.0
        # 用姿态模型作为"主"人体检测器（而不是只在 YOLO-World 空手时才兜底）。
        # 原因：YOLO-World 对躺姿几乎检不出来，而姿态模型能看到。
        # 打开后可以停掉 yoloworld 节点，省一个 GPU 进程（显存更安全）。
        self.pose_first_detection = bool(
            rospy.get_param("~pose_first_detection", True)
        )
        # 最近一次姿态检测到的人体框，供窗口显示
        self._last_pose_boxes = []
        # 整帧检测时"顺带"得到的姿态结论（lying/sitting/upright/空）
        self._last_pose_verdict = ""
        # ---- 取景体检：采集前判断"人是否完整、比例合适地落在画面里" ----
        # 用"人体框高度占画面的比例"而不是"离我几米"，高矮都能自适应：
        # 个子高就自然站远一点，个子矮就站近一点，画面效果一致。
        self.framing_check_enabled = bool(
            rospy.get_param("~framing_check_enabled", True)
        )
        self.framing_check_samples = max(
            1, int(rospy.get_param("~framing_check_samples", 5))
        )
        self.framing_check_interval = max(
            0.02, float(rospy.get_param("~framing_check_interval", 0.15))
        )
        self.framing_max_attempts = max(
            1, int(rospy.get_param("~framing_max_attempts", 3))
        )
        # 人脸阶段：要脸够大 -> 人体占画面至少一半
        self.framing_face_min_height_ratio = float(
            rospy.get_param("~framing_face_min_height_ratio", 0.50)
        )
        self.framing_face_max_height_ratio = float(
            rospy.get_param("~framing_face_max_height_ratio", 0.98)
        )
        # 身形阶段：要全身入镜 -> 框完整且不贴边
        self.framing_body_min_height_ratio = float(
            rospy.get_param("~framing_body_min_height_ratio", 0.45)
        )
        self.framing_body_max_height_ratio = float(
            rospy.get_param("~framing_body_max_height_ratio", 0.95)
        )
        # 人体框贴到画面上下边缘算"太近"（说明没拍全）
        self.framing_edge_margin = max(
            0.0, min(0.2, float(rospy.get_param("~framing_edge_margin", 0.02)))
        )
        self.framing_far_text = rospy.get_param(
            "~framing_far_text", "有点远，请再靠近一步。"
        )
        self.framing_near_text = rospy.get_param(
            "~framing_near_text", "有点近，请往后退半步。"
        )
        self.framing_no_person_text = rospy.get_param(
            "~framing_no_person_text", "我没有看到您，请站到镜头前面来。"
        )
        # 人脸阶段要不要播报这类"太远/太近/没看到您"的提示。
        # 默认关掉：人脸阶段已经有一句明确的站位提示
        # （"请距我半米，把脸正对着我上方的摄像头"），再按阈值播报反而啰嗦。
        # 身形阶段保持原样（那里的提示有用）。
        self.framing_speak_face = bool(
            rospy.get_param("~framing_speak_face", False)
        )
        # ---- 分步播报话术：先说清"第几步、离多远" ----
        self.face_announce_text = rospy.get_param(
            "~face_announce_text",
            "%s，第一步人脸注册。请距我半米，把脸正对着我上方的摄像头。",
        )
        self.body_announce_text = rospy.get_param(
            "~body_announce_text",
            "第二步身形注册。请退后让我看到全身，先正对着我。",
        )
        self.body_front_prompt = rospy.get_param(
            "~body_front_prompt", "正在记录正面，请保持不动。"
        )
        self.body_side_prompt = rospy.get_param(
            "~body_side_prompt", "请侧身，正在记录侧面。"
        )
        # 身形分两段：先正面，再侧面（合计样本数与原来一致：10 + 6 = 16）
        self.body_front_sample_count = max(
            1, int(rospy.get_param("~body_front_sample_count", 10))
        )
        self.body_front_seconds = max(
            0.5, float(rospy.get_param("~body_front_seconds", 4.5))
        )
        self.body_side_sample_count = max(
            1, int(rospy.get_param("~body_side_sample_count", 6))
        )
        self.body_side_seconds = max(
            0.5, float(rospy.get_param("~body_side_seconds", self.record_seconds))
        )
        # 覆盖父类的采集姿态：把"正面转侧面"换成明确的"先正面、再侧面"
        self.owner_record_poses = [
            ("front", "正面", self.body_front_prompt),
            ("side", "侧面", self.body_side_prompt),
        ]
        # 人脸阶段为 True（save_crop 会顺带提取人脸）；
        # 身形阶段临时置 False，避免身形采集中重复提取人脸。
        self._collect_face_during_body = True

    # ------------------------------------------------------------------
    # 姓名 + 确认
    # ------------------------------------------------------------------
    def parse_owner_name(self, answer):
        """沿用父类解析，再把阿拉伯数字规整成中文（'张3' -> '张三'）。"""
        name = super().parse_owner_name(answer)
        if not name or not self.normalize_name_digits:
            return name
        normalized = "".join(
            self.digit_to_chinese.get(ch, ch) for ch in name
        )
        if normalized != name:
            rospy.loginfo("姓名数字规整：%s -> %s", name, normalized)
        return normalized

    @staticmethod
    def parse_owner_confirmation(text):
        """解析"确认 / 重来"回答。返回 True / False / None。

        逻辑与 task1_find_owner_real.py 的 parse_owner_confirmation 保持一致，
        包括对"机器人自己提示语回声"的排除（提示语结尾也带着"确认/重来"）。

        这里额外做了两件事（实测踩坑后补的）：
          1. 同音误听归一化：ASR 把"重来"听成"从来"，只认固定写法就会
             一句话也认不出来，白白卡满一个超时。
          2. 改成"关键词包含"判定而不是只认整句相等：用户说
             "对的对的""那就重来吧"这种自然说法也能认。
        否定词先判：否则"不对"会因为含"对"被当成确认。
        """
        import re

        compact = re.sub(r"\s+", "", str(text or ""))
        compact = re.sub(r"[，。！？,.!?、；;：:“”\"'()（）]", "", compact)
        if not compact:
            return None
        for echo_marker in (
            "您说的是",
            "不对就说错误",
            "不对就说错了",
            "对就说确认",
            "不对就说重来",
            "正确请回答确认",
            "不正确请回答重来",
            "请确认这个名字",
            "请回答确认",
            "请回答重来",
        ):
            if echo_marker in compact:
                return None
        if len(compact) > 8:
            return None
        # 1) 同音误听归一化（顺序放在判定之前）
        for wrong, right in (
            ("从来", "重来"),
            ("虫来", "重来"),
            ("重赖", "重来"),
            ("中来", "重来"),
            ("冲来", "重来"),
            ("不队", "不对"),
            ("不事", "不是"),
            ("缺认", "确认"),
            ("确人", "确认"),
            ("却认", "确认"),
            ("确任", "确认"),
        ):
            compact = compact.replace(wrong, right)
        # 极短的回答（ASR 只吐出一两个字）单独认
        if compact in ("不", "错", "否", "重", "改"):
            return False
        # 2) 先否定、后肯定（"不对"里带着"对"，顺序反了就会误判）
        if any(
            marker in compact
            for marker in (
                "不是",
                "不对",
                "错误",
                "错了",
                "错啦",
                "重来",
                "重新",
                "重说",
                "再说",
                "再来",
                "换一个",
                "不确定",
                "不清楚",
                "不知道",
                "不可以",
                "不好",
                "不行",
                "不要",
                "否",
            )
        ):
            return False
        if any(
            marker in compact
            for marker in (
                "确认",
                "确定",
                "没错",
                "正确",
                "对的",
                "对",
                "是的",
                "是",
                "好的",
                "好",
                "可以",
            )
        ):
            return True
        return None

    def wait_for_owner_confirmation(self):
        """等一句回答并解析成"确认/否认"。超时返回 None。"""
        unparsed_count = 0
        hint_spoken = False
        with self.name_condition:
            self.latest_name_answer = None
            self.accepting_name_answer = True
            deadline = time.time() + max(3.0, float(self.name_answer_timeout))
            while not rospy.is_shutdown() and time.time() < deadline:
                answer = self.latest_name_answer
                if answer:
                    self.latest_name_answer = None
                    decision = self.parse_owner_confirmation(answer)
                    if decision is not None:
                        self.accepting_name_answer = False
                        rospy.loginfo(
                            "Owner %d name confirmation: %s raw=%s",
                            self.current_owner_index,
                            "确认" if decision else "重来",
                            answer,
                        )
                        return decision
                    # 听到了、但认不出是"确认"还是"重来"：以前是默默丢掉，
                    # 结果白白干等一整个超时。这里连着两次认不出就提示一次，
                    # 让用户知道该怎么答（提示语用和原提示一样的说法，
                    # 这样它自己的回声会被回声过滤器挡住）。
                    unparsed_count += 1
                    rospy.logwarn(
                        "认不出这是确认还是重来（第 %d 次）：%s", unparsed_count, answer
                    )
                    if unparsed_count >= 2 and not hint_spoken:
                        hint_spoken = True
                        self.speak(
                            "我没有听清。对就说确认，不对就说错误。", wait=False
                        )
                # 等确认期间也要刷新调试窗口，否则窗口会"卡住不动"
                self.update_yolo_window("等待姓名确认")
                self.name_condition.wait(timeout=0.2)
            self.accepting_name_answer = False
        return None

    def wait_for_confirmed_owner_name(self, owner_index):
        """问姓名 -> 复述并要求确认，直到确认或重试次数用尽。

        返回：确认后的姓名 / None（第 2 位起说了"跳过"）/ ""（中断或彻底失败）
        """
        # 下一轮用哪句提示：第一次用标准长提示（带格式例子），
        # 之后用短提示——听到"重来"就一句"好的，请再说一次您的名字"直接说，
        # 不再单独播报一遍致歉语。
        next_prompt = None
        for attempt in range(self.enroll_retries):
            name = self.wait_for_owner_name(owner_index, prompt_text=next_prompt)
            if name is None:
                return None
            if not name:
                return ""
            self.speak(self.confirm_prompt_template % name, wait=True)
            decision = self.wait_for_owner_confirmation()
            if decision is True:
                return name
            if decision is None:
                rospy.logwarn(
                    "Owner %d name confirmation timed out (attempt %d/%d)",
                    owner_index,
                    attempt + 1,
                    self.enroll_retries,
                )
                next_prompt = self.name_timeout_prompt_text
                continue
            next_prompt = self.name_retry_prompt_text
        self.speak(self.name_failed_text, wait=True)
        return ""

    # ------------------------------------------------------------------
    # 第 3 步：只采人脸
    # ------------------------------------------------------------------
    def speak_with_live_window(self, text, extra_seconds=0.0):
        """一边播报一边持续刷新调试窗口。

        为什么需要：`speak(wait=True)` 是纯 sleep，这 2~3 秒里没有任何
        update_yolo_window 调用，窗口就停在上一帧。注册时"人脸阶段结束 →
        身形阶段开始"中间正好有这么一段播报，人听到"请退后"就开始退，
        但画面不动，结果退过头（实测）。改成边播边刷新即可。
        """
        text = str(text or "")
        if not text:
            return
        cps = max(0.1, float(getattr(self, "tts_chars_per_second", 6.0)))
        wait_seconds = (
            max(float(getattr(self, "tts_min_wait", 1.0)), len(text) / cps)
            + float(getattr(self, "tts_extra_wait", 0.4))
            + max(0.0, float(extra_seconds))
        )
        self.speak(text, wait=False)
        deadline = time.time() + wait_seconds
        while not rospy.is_shutdown() and time.time() < deadline:
            self.update_yolo_window("播报中…")
            rospy.sleep(0.1)

    def measure_framing(self):
        """连取几帧，返回 (人体框高度占比中位数, 是否贴上下边缘, 有效样本数)。

        返回 (None, ..., 0) 表示这几帧里一个人都没检测到。
        """
        heights = []
        touching_edge = False
        for _ in range(max(1, self.framing_check_samples)):
            image, detections = self.snapshot()
            if image is not None:
                candidates = self.person_candidates(image, detections)
                if candidates:
                    candidate = candidates[0]
                    heights.append(float(candidate.get("height_ratio", 0.0)))
                    bbox = candidate.get("bbox") or [0, 0, 0, 0]
                    frame_height = float(image.shape[0])
                    if (
                        bbox[1] <= frame_height * self.framing_edge_margin
                        or bbox[3] >= frame_height * (1.0 - self.framing_edge_margin)
                    ):
                        touching_edge = True
            # 体检期间也要刷新调试窗口：否则"人脸阶段结束 -> 身形阶段开始"
            # 这段窗口是停住的，人已经开始往后退了、画面却不动，容易退过头。
            self.update_yolo_window("取景体检中")
            rospy.sleep(self.framing_check_interval)
        if not heights:
            return None, touching_edge, 0
        heights.sort()
        return heights[len(heights) // 2], touching_edge, len(heights)

    def ensure_framing(self, stage):
        """采集前的"取景体检"：太远/太近/没人都提示对方调整。

        stage = "face"  -> 人脸阶段，要求人体占画面一半以上（脸才够大）
        stage = "body"  -> 身形阶段，要求全身完整入镜且不贴边
        返回 True 表示取景合格；False 表示重试用尽（仍会继续采集，不卡死）。
        """
        if not self.framing_check_enabled or not self.pose_first_detection:
            return True
        if stage == "face":
            min_ratio = self.framing_face_min_height_ratio
            max_ratio = self.framing_face_max_height_ratio
        else:
            min_ratio = self.framing_body_min_height_ratio
            max_ratio = self.framing_body_max_height_ratio

        # 人脸阶段默认只记日志、不播报（提示语已经单独说过一次站位要求）；
        # 身形阶段照旧播报。
        speak_hints = self.framing_speak_face if stage == "face" else True

        for attempt in range(1, self.framing_max_attempts + 1):
            ratio, touching_edge, count = self.measure_framing()
            if ratio is None:
                rospy.logwarn(
                    "取景体检（%s）第 %d/%d 次：没检测到人",
                    stage,
                    attempt,
                    self.framing_max_attempts,
                )
                if speak_hints:
                    self.speak(self.framing_no_person_text, wait=True)
                continue
            if ratio < min_ratio:
                rospy.loginfo(
                    "取景体检（%s）第 %d/%d 次：太远（高度占比 %.2f < %.2f）",
                    stage,
                    attempt,
                    self.framing_max_attempts,
                    ratio,
                    min_ratio,
                )
                if speak_hints:
                    self.speak(self.framing_far_text, wait=True)
                continue
            if ratio > max_ratio or touching_edge:
                rospy.loginfo(
                    "取景体检（%s）第 %d/%d 次：太近（高度占比 %.2f / 贴边=%s）",
                    stage,
                    attempt,
                    self.framing_max_attempts,
                    ratio,
                    touching_edge,
                )
                if speak_hints:
                    self.speak(self.framing_near_text, wait=True)
                continue
            rospy.loginfo(
                "取景体检（%s）：合格（高度占比 %.2f，样本 %d）",
                stage,
                ratio,
                count,
            )
            return True
        rospy.logwarn("取景体检（%s）重试用尽，按当前取景继续采集", stage)
        return False

    def capture_owner_face_only(self):
        """只采集人脸特征，返回人脸样本列表。"""
        if not self.face_model_ready or self.face_app is None:
            raise RuntimeError("人脸模型未就绪，无法采集人脸特征")

        self.current_record_face_embeddings = []
        self.current_record_pose_id = "face"
        self.current_record_pose_name = "face"
        sample_dir = self.profile_sample_dir()

        # 边播报边刷新窗口：这句说完马上就要体检/采集，窗口不能停
        self.speak_with_live_window(self.face_announce_text % self.owner_name)
        # 取景体检：不合适就先提示对方调整，再开始采集
        self.ensure_framing("face")
        self.play_ding()

        deadline = time.time() + self.face_capture_seconds
        next_sample_time = 0.0
        rate = rospy.Rate(30)
        while (
            not rospy.is_shutdown()
            and time.time() < deadline
            and len(self.current_record_face_embeddings) < self.face_sample_count
        ):
            now = time.time()
            if now < next_sample_time:
                self.update_yolo_window("登记人脸")
                rate.sleep()
                continue
            next_sample_time = now + self.face_sample_interval

            image, detections = self.snapshot()
            if image is None:
                rospy.logwarn_throttle(1.0, "等待相机图像（%s）", self.image_topic)
                self.update_yolo_window("等待相机")
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if not candidates:
                rospy.logwarn_throttle(
                    1.0,
                    "等待人体检测结果（姿态模型；YOLO-World 话题=%s）",
                    self.detections_topic,
                )
                self.update_yolo_window("等待人体检测")
                rate.sleep()
                continue
            crop, _crop_bbox = self.crop_candidate(image, candidates[0])
            if crop is None:
                rate.sleep()
                continue

            index = len(self.current_record_face_embeddings) + 1
            before = len(self.current_record_face_embeddings)
            # save_crop 内部会做人脸提取并把结果记进 current_record_face_embeddings
            self.save_crop(crop, sample_dir, "owner_face", index)
            if len(self.current_record_face_embeddings) > before:
                rospy.loginfo(
                    "Owner %d face sample %d captured", self.current_owner_index, index
                )
            self.publish_status(
                "face_recording_sample",
                owner_index=self.current_owner_index,
                name=self.owner_name,
                sample=index,
                samples=len(self.current_record_face_embeddings),
                target=self.face_sample_count,
            )
            self.update_yolo_window("登记人脸")
            rate.sleep()

        captured = len(self.current_record_face_embeddings)
        self.publish_status(
            "face_recording_done",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            samples=captured,
        )
        if captured < self.face_min_samples:
            self.speak(self.record_failed_text, wait=True)
            raise RuntimeError(
                "only captured %d/%d usable face samples for owner %d"
                % (captured, self.face_min_samples, self.current_owner_index)
            )
        rospy.loginfo(
            "Owner %d face enrollment done: %d samples", self.current_owner_index, captured
        )
        return list(self.current_record_face_embeddings)

    # ------------------------------------------------------------------
    # 第 4 步：只采身形
    # ------------------------------------------------------------------
    def capture_owner_body_only(self):
        """只采集身形（ReID）特征，返回 (embeddings, color_embeddings, sample_meta)。"""
        self._collect_face_during_body = False
        try:
            # 先讲清楚：这是第二步、要站一米多、先正面后侧面
            # 这句"请退后…"尤其关键：说完人会立刻后退，窗口必须跟着动，
            # 否则画面停在上一帧、人会退过头（实测）
            self.speak_with_live_window(self.body_announce_text)
            # 身形阶段要求全身入镜，先体检一次
            self.ensure_framing("body")
            timing = {
                "front": (self.body_front_seconds, self.body_front_sample_count),
                "side": (self.body_side_seconds, self.body_side_sample_count),
            }
            all_embeddings = []
            all_color_embeddings = []
            all_sample_meta = []
            for pose_id, pose_name, prompt_text in self.owner_record_poses:
                duration, sample_count = timing.get(pose_id, (None, None))
                embeddings, color_embeddings, sample_meta = self.record_owner_pose(
                    pose_id,
                    pose_name,
                    prompt_text,
                    len(all_embeddings),
                    duration=duration,
                    sample_count=sample_count,
                )
                all_embeddings.extend(embeddings)
                all_color_embeddings.extend(color_embeddings)
                all_sample_meta.extend(sample_meta)
            return all_embeddings, all_color_embeddings, all_sample_meta
        finally:
            self._collect_face_during_body = True

    def save_crop(self, crop, directory, prefix, index):
        """身形阶段跳过人脸提取，避免把身形帧的人脸混进档案。"""
        if not self._collect_face_during_body:
            return PersonReidOwnerTest.save_crop(self, crop, directory, prefix, index)
        return super().save_crop(crop, directory, prefix, index)

    # ------------------------------------------------------------------
    # 躺姿判定：父类只认"宽 > 高×1.25"，迎面拍到的躺姿会漏判
    # ------------------------------------------------------------------
    def wait_for_camera_inputs(self):
        """姿态检测为主时，只要相机有图就能启动，不再强制要求"行人检测"话题。

        父类要求 image 和 detections 都就绪才放行；但本脚本可以用姿态模型
        自己做人体检测（YOLO-World 对躺姿几乎检不出来），所以把 yoloworld
        节点停掉时也必须能正常启动。
        """
        if not self.pose_first_detection:
            return super().wait_for_camera_inputs()
        started = time.time()
        deadline = started + self.startup_timeout
        last_report = 0.0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                has_image = self.latest_image is not None
            if has_image:
                self.publish_status("camera_ready")
                return True
            # 每 5 秒提示一次，避免"静默等 35 秒然后报错"，看不出卡在哪
            if time.time() - last_report >= 5.0:
                last_report = time.time()
                rospy.logwarn(
                    "等待相机图像：%s（已等 %.0f 秒 / %.0f 秒）",
                    self.image_topic,
                    time.time() - started,
                    self.startup_timeout,
                )
            rate.sleep()
        raise RuntimeError(
            "相机没有出图：%s（已等 %.0f 秒）。请依次检查："
            "① 机器人栈是否在跑（robot.sh status）；"
            "② rostopic hz %s 有没有数据；"
            "③ 是否有残留进程占着 Kinect（先跑 robot.sh stop 再起栈）"
            % (self.image_topic, time.time() - started, self.image_topic)
        )

    def init_yolo_window(self):
        """用 WINDOW_AUTOSIZE 建调试窗口。

        父类用的是 cv2.WINDOW_NORMAL（可自由缩放），实测在机器人上会显示成
        一整块黑框、看不到画面；AUTOSIZE 会按图像尺寸自动适配，稳定得多。
        """
        if not self.show_yolo_window or self.yolo_window_failed:
            return
        try:
            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_AUTOSIZE)
            self.yolo_window_ready = True
            rospy.loginfo("调试窗口已开启（AUTOSIZE）：%s", self.yolo_window_name)
        except Exception as exc:
            self.yolo_window_ready = False
            self.yolo_window_failed = True
            rospy.logwarn("无法创建调试窗口：%s", exc)

    def update_yolo_window(self, status_text=""):
        """调试窗口：绿色 = YOLO-World 检到的框，青色 = 姿态模型检到的框。

        这里整段自己实现（不再调用父类版本），原因是：父类版本里已经做过
        一次 imshow + waitKey，如果再叠一次，OpenCV 的 GTK 事件循环容易被
        拖住、窗口表现为"卡住不刷新"——实测就是这样。一次 imshow 画全。
        """
        if not self.yolo_window_ready:
            return
        with self.lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()
            detections = list(self.latest_detections)
        height, width = image.shape[:2]

        # 绿色：YOLO-World（如果它还在跑）
        for detection in detections:
            label = str(getattr(detection, "class_name", "person") or "person")
            score = float(getattr(detection, "score", 0.0) or 0.0)
            x1 = max(0, min(width - 1, int(getattr(detection, "xmin", 0))))
            y1 = max(0, min(height - 1, int(getattr(detection, "ymin", 0))))
            x2 = max(0, min(width - 1, int(getattr(detection, "xmax", 0))))
            y2 = max(0, min(height - 1, int(getattr(detection, "ymax", 0))))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                image,
                "%s %.2f" % (label, score),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

        # 青色：姿态模型（现在的人体检测主力）
        for candidate in self._last_pose_boxes:
            bbox = candidate.get("bbox") or [0, 0, 0, 0]
            x1 = max(0, min(width - 1, int(bbox[0])))
            y1 = max(0, min(height - 1, int(bbox[1])))
            x2 = max(0, min(width - 1, int(bbox[2])))
            y2 = max(0, min(height - 1, int(bbox[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                image,
                "pose %.2f" % float(candidate.get("score", 0.0)),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        header = "pose=%d  yolo=%d" % (len(self._last_pose_boxes), len(detections))
        if status_text:
            header += " | " + status_text
        cv2.putText(
            image,
            header,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        try:
            cv2.imshow(self.yolo_window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rospy.signal_shutdown("YOLO debug window closed by user")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "调试窗口更新失败：%s", exc)
            self.yolo_window_ready = False

    def person_candidates(self, image, detections):
        """放宽"人"类标签的过滤，并补上"框高占画面比例"。

        父类只接受 class_name 完全等于 'person' 的检测框。但 YOLO-World 是
        开放词表检测器，类别名就是提示词；给躺姿加 "lying person" 之类的
        提示词能显著提高检出率，而那些标签不等于 'person'，会被父类丢掉。
        这里放宽为：只要标签里包含 person 就接受。
        """
        filtered = []
        for detection in detections:
            label = str(getattr(detection, "class_name", "") or "").strip().lower()
            if label and "person" not in label:
                continue
            filtered.append(detection)
        candidates = super().person_candidates(image, filtered)
        image_height = float(max(1, image.shape[0]))
        for candidate in candidates:
            bbox = candidate.get("bbox") or [0, 0, 0, 0]
            candidate["height_ratio"] = (
                float(bbox[3]) - float(bbox[1])
            ) / image_height

        if not self.pose_first_detection:
            # 老行为：YOLO-World 为主，它空手时才用姿态兜底
            if not candidates:
                candidates = self.pose_person_candidates(image)
            return candidates

        # 新行为：姿态模型为主（对躺姿更准），YOLO-World 只当兜底。
        # 这样即使把 yoloworld 节点停掉，找人也照常工作。
        pose_candidates = self.pose_person_candidates(image)
        if pose_candidates:
            return pose_candidates
        return candidates

    def pose_person_candidates(self, image):
        """用姿态模型在整帧上找人体，返回与 YOLO-World 同格式的候选列表。

        只在 YOLO-World 什么都没检到时调用（见 person_candidates）。
        """
        if not self.pose_fallback_enabled:
            return []
        analyzer = self.pose_analyzer
        if analyzer is None or not analyzer.ready or analyzer.model is None:
            return []
        now = time.time()
        # 只有"姿态只是兜底"时才限流。姿态当主检测器时必须每轮都跑：
        # 一旦被限流跳过，这一轮就没有任何候选 -> 连续命中计数被扣分 ->
        # 永远凑不满"连续 N 次"（实测就是第一次失败、第二次侥幸成功的原因）。
        if (
            not self.pose_first_detection
            and now - self._last_pose_fallback_time < self.pose_fallback_interval
        ):
            return []
        self._last_pose_fallback_time = now
        self._last_pose_boxes = []
        self._last_pose_verdict = ""
        try:
            results = analyzer._predict(image)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "姿态兜底检测失败：%s", exc)
            return []
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        # 同一次推理里顺便拿到"每个人各自的"姿态结论：
        # 检测人体时关键点已经算好了，不必再裁图重跑；而且要按人分开，
        # 否则多人在场时会拿"模型最有把握的那个人"的姿态去判断当前候选，
        # 结果就是同一个躺姿被判成 lying/sitting 来回跳。
        keypoints = []
        keypoint_confidences = []
        if getattr(results[0], "keypoints", None) is not None:
            try:
                keypoints = analyzer._tensor_to_list(results[0].keypoints.xy)
                keypoint_confidences = analyzer._tensor_to_list(
                    results[0].keypoints.conf
                )
            except Exception:
                keypoints = []
                keypoint_confidences = []

        def pose_label_for(index, box, score):
            if index >= len(keypoints) or index >= len(keypoint_confidences):
                return ""
            try:
                feature = analyzer._feature(
                    {
                        "keypoints": keypoints[index],
                        "confidences": keypoint_confidences[index],
                        "box": box,
                        "box_confidence": float(score),
                    },
                    image.shape,
                )
            except Exception:
                return ""
            if feature.get("lying_like"):
                return "lying"
            if feature.get("sitting_like"):
                return "sitting"
            if feature.get("upright_like"):
                return "upright"
            return ""
        try:
            xyxy = analyzer._tensor_to_list(boxes.xyxy)
            confidences = analyzer._tensor_to_list(boxes.conf)
        except Exception:
            return []

        height, width = image.shape[:2]
        image_area = float(max(1, height * width))
        candidates = []
        for index, (box, score) in enumerate(zip(xyxy, confidences)):
            x1 = max(0, min(width - 1, int(round(float(box[0])))))
            y1 = max(0, min(height - 1, int(round(float(box[1])))))
            x2 = max(0, min(width - 1, int(round(float(box[2])))))
            y2 = max(0, min(height - 1, int(round(float(box[3])))))
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
            if area_ratio < self.detection_min_area_ratio:
                continue
            candidates.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                    "area_ratio": area_ratio,
                    "aspect_ratio": float(x2 - x1) / max(1.0, float(y2 - y1)),
                    "priority": float(score) * area_ratio,
                    "height_ratio": (y2 - y1) / float(height),
                    "pose_label": pose_label_for(index, box, score),
                }
            )
        if candidates:
            candidates.sort(key=lambda item: item["priority"], reverse=True)
            self._last_pose_boxes = candidates[: self.top_k_candidates]
            rospy.loginfo(
                "姿态检测在整帧上找到 %d 个人"
                "（分数 %s，最高那个的宽高比=%.2f 高度占比=%.2f）",
                len(candidates),
                ", ".join("%.2f" % item["score"] for item in candidates[:3]),
                candidates[0]["aspect_ratio"],
                candidates[0]["height_ratio"],
            )
        return candidates[: self.top_k_candidates]

    def is_lying_candidate(self, candidate):
        """躺姿判定：父类判据之外，再补一条"框很短"的判据。

        躺在地上/床上若被人从头顶方向拍到，人体框会又短又不宽（宽高比
        到不了 1.25），父类判不出躺姿，于是落进"严格人脸判定"被误拒——
        这正是"躺着的主人找不到"的原因。

        补判：框高度占画面比例很小、且形状不是细长条 → 也按躺姿处理。
        按躺姿处理后会走宽松的 ReID 通道（阈值 0.6），不要求人脸。
        """
        if super().is_lying_candidate(candidate):
            return True
        aspect = float(candidate.get("aspect_ratio", 0.0) or 0.0)
        height_ratio = float(candidate.get("height_ratio", 1.0) or 1.0)
        lying = (
            height_ratio <= self.lying_extra_height_ratio
            and aspect >= self.lying_extra_aspect_min
        )
        rospy.loginfo_throttle(
            2.0,
            "候选人体框：宽高比=%.2f 高度占比=%.2f -> %s",
            aspect,
            height_ratio,
            "按躺姿处理(走宽松ReID通道)" if lying else "按站立处理(以脸为准)",
        )
        return lying

    # ------------------------------------------------------------------
    # 判定：给"陌生人"一条明确的拒绝路径
    # ------------------------------------------------------------------
    def margin_ok(self, result):
        """多人时要求"第一名与第二名的分差"够大，否则说明分不清，宁可不认。"""
        margin = result.get("owner_score_margin")
        if (
            len(self.owner_profiles) > 1
            and margin is not None
            and float(margin) < self.owner_score_margin_threshold
        ):
            return False
        return True

    def strong_lying_geometry(self, result):
        """人体框是不是"明确横躺"（宽 ≥ 高 strong_lying_aspect 倍）。

        用它来纠正姿态判决的抖动：站着的人框是竖的，坐着大概方，只有躺着的
        才是这种又宽又扁的形状。
        """
        candidate = (result or {}).get("candidate") or {}
        bbox = candidate.get("bbox") or [0, 0, 0, 0]
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if height <= 1.0 or width <= 0.0:
            return False
        return (width / height) >= self.strong_lying_aspect

    def result_is_match(self, result, default_threshold):
        """两步判定：先常规，被拒之后才用单帧姿态确认是不是躺着/坐着。

        为什么分两步：站着的人（主人或陌生人）用常规人脸判定就够了，
        没必要对每个候选都跑姿态推理；只有"本来要被拒"的候选才多花这一次
        推理，去确认它是不是躺着——躺着的人看不到正脸，只能改走外观通道。
        """
        if result is None:
            return False
        if not self.margin_ok(result):
            rospy.loginfo_throttle(2.0, "拒绝：多位主人的分数太接近，分不清是谁")
            return False

        # 第 1 步：常规判定（人脸为准；已被认成躺姿的走父类宽松通道）
        if self.strict_channel_match(result, default_threshold):
            return True

        # 第 2 步：常规判定没通过，才跑一次单帧姿态
        pose_label = self.classify_candidate_pose(result)
        # 姿态判决会抖：同一个人同一段视频里 sitting / lying / upright 来回跳，
        # 而"坐"和"躺"用的门槛差了 0.15。实测出现过"同一个分数 0.674，
        # 判 lying 就接受、判 sitting 就被拒"这种自相矛盾。
        # 所以当"人体框形状"明确是横躺（宽 ≥ 高 strong_lying_aspect 倍）时，
        # 不采信抖动的 sitting 判决，直接按躺姿通道走。
        if pose_label != "upright" and self.strong_lying_geometry(result):
            if pose_label != "lying":
                rospy.loginfo(
                    "单帧姿态=%s，但人体框形状明显是横躺 -> 按躺姿通道复核",
                    pose_label,
                )
            pose_label = "lying"
        if pose_label == "lying":
            # 躺着看不到正脸是正常的 → 允许只看外观
            rospy.loginfo(
                "常规判定未通过，但单帧姿态=lying -> 改走外观通道复核"
            )
            return self.appearance_match(
                result, tag="外观通道复核(lying)"
            )
        if pose_label == "sitting":
            # 坐着的人是能看到正脸的。所以：
            #   检测到人脸但不像  -> 直接拒绝（坐着的陌生人就是这么混进来的）
            #   完全没检测到人脸  -> 才允许外观通道，但阈值要提高
            face_score = result.get("face_score")
            if face_score is not None:
                rospy.loginfo(
                    "拒绝：姿态=sitting 且检测到人脸（相似度 %.3f）但未通过 -> 不认",
                    float(face_score),
                )
                return False
            rospy.loginfo(
                "姿态=sitting 且没有人脸 -> 外观通道复核（阈值提高到 %.2f）",
                self.sitting_appearance_threshold,
            )
            return self.appearance_match_with(
                result,
                tag="外观通道复核(sitting)",
                threshold=self.sitting_appearance_threshold,
            )
        rospy.loginfo("常规判定未通过，单帧姿态=%s -> 维持拒绝", pose_label)
        return False

    def init_pose_helper(self):
        """加载单帧姿态模型（躺/坐判断用）。建议在开始扫描前调用。"""
        if self.pose_analyzer is not None:
            return True
        core_dir = _find_action_core_dir()
        if not core_dir:
            rospy.logwarn("找不到动作识别核心目录，躺姿判断将不可用")
            return False
        try:
            spec = importlib.util.spec_from_file_location(
                "owner_test_action_core",
                os.path.join(core_dir, "qwen_action_recognition_node.py"),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            rospy.logwarn("加载动作识别核心失败：%s", exc)
            return False

        model_path = module.PoseActionAnalyzer.resolve_model_path(
            rospy.get_param("~pose_model_path", "")
        )
        # ⚠️ 扫描时的单帧姿态判断默认走 CPU，不要用 cuda！
        # GTX 1650 只有 4GB，YOLO-World（行人检测）已经占着一个 CUDA context，
        # 本脚本再建一个会把显存挤爆 —— 实测后果是 GPU 掉线，之后连
        # yoloworld_async 都起不来（报 torch.cuda.is_available() is false）。
        # 单帧推理只在"身份判定失败"时才跑（一次扫描总共几次），
        # CPU 上 100~300ms 完全可以接受；真要上 GPU 再显式传
        # _scan_pose_device:=cuda:0。
        scan_pose_device = rospy.get_param("~scan_pose_device", "cpu")
        analyzer = module.PoseActionAnalyzer(
            model_path,
            scan_pose_device,
            max(160, int(rospy.get_param("~pose_image_size", 416))),
            float(rospy.get_param("~pose_confidence", 0.25)),
            float(rospy.get_param("~pose_iou", 0.45)),
            max(1, int(rospy.get_param("~pose_max_detections", 4))),
            # CPU 上不能用半精度
            bool(rospy.get_param("~pose_half", True))
            and str(scan_pose_device).startswith("cuda"),
        )
        ok, message = analyzer.initialize()
        if not ok:
            rospy.logwarn("单帧姿态模型加载失败：%s", message)
            return False
        self.pose_core = module
        self.pose_analyzer = analyzer
        rospy.loginfo(
            "躺姿判断用的姿态模型已就绪：model=%s device=%s（%s）",
            model_path,
            analyzer.device,
            message,
        )
        return True

    def classify_candidate_pose(self, result):
        """对候选裁剪图跑单帧姿态，返回 'lying' / 'sitting' / 'upright' / 'unknown'。

        躺姿的人体框又宽又扁，而 YOLO-Pose 是按直立人体训练的，
        直接对原图推理经常"检不到人"（返回 unknown，等于辅助失效）。
        所以：原图检不到人时，再把裁剪图旋转 ±90° 各试一次——
        躺姿转过来就接近直立姿态，模型就能检到了。

        注意顺序：**只有原图检不到人时才试旋转版**。
        因为把直立的人旋转 90° 也会"看起来像躺着"，反过来用会误判，
        所以原图能检到人时以原图为准，绝不尝试旋转。
        """
        analyzer = self.pose_analyzer
        if analyzer is None or not analyzer.ready:
            return "unknown"

        # 优先用"整帧检测人体"时同一次推理得到的姿态结论：
        # 那次推理本来就是为了找人体框，关键点已经算好了，直接拿来用最可靠
        # （裁剪图又宽又扁，单独对裁剪图推理经常检不到人）。
        # 优先：按"候选框"去匹配姿态检测里的同一个人。
        # 多人在场时必须这样——否则拿的是"模型最有把握那个人"的姿态，
        # 会把躺着的你判成别人的坐姿。
        target_bbox = (result.get("candidate") or {}).get("bbox")
        if target_bbox and len(target_bbox) >= 4:
            try:
                key = tuple(int(round(float(v))) for v in target_bbox[:4])
            except (TypeError, ValueError):
                key = None
            if key is not None:
                for candidate in self._last_pose_boxes:
                    bbox = candidate.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    if tuple(int(v) for v in bbox[:4]) != key:
                        continue
                    label = str(candidate.get("pose_label") or "")
                    if label:
                        rospy.loginfo(
                            "单帧姿态(同一次推理, 按候选框匹配) -> %s", label
                        )
                        return label

        if self._last_pose_verdict:
            rospy.loginfo(
                "单帧姿态(整帧最佳那个人) -> %s", self._last_pose_verdict
            )
            return self._last_pose_verdict

        crop = result.get("crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            return "unknown"

        def verdict(feature):
            if feature.get("lying_like"):
                return "lying"
            if feature.get("sitting_like"):
                return "sitting"
            if feature.get("upright_like"):
                return "upright"
            return "unknown"

        # 第一步：原图
        try:
            features = analyzer.extract_features([crop])
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "单帧姿态判断失败(原始)：%s", exc)
            features = []
        if features:
            feature = features[-1]
            label = verdict(feature)
            rospy.loginfo(
                "单帧姿态(原始)：宽高比=%.2f 躯干竖直度=%.2f -> %s",
                float(feature.get("aspect", 0.0)),
                float(feature.get("torso_verticality", 0.0)),
                label,
            )
            return label

        # 第二步：原图检不到人（典型情况就是躺着），再试旋转 ±90°
        variants = []
        try:
            variants = [
                ("rot90_cw", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
                ("rot90_ccw", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "候选图旋转失败：%s", exc)

        for variant_name, image in variants:
            try:
                features = analyzer.extract_features([image])
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "单帧姿态判断失败(%s)：%s", variant_name, exc)
                continue
            if not features:
                continue
            feature = features[-1]
            label = verdict(feature)
            rospy.loginfo(
                "单帧姿态(%s，原图检不到人)：宽高比=%.2f 躯干竖直度=%.2f -> %s",
                variant_name,
                float(feature.get("aspect", 0.0)),
                float(feature.get("torso_verticality", 0.0)),
                label,
            )
            if label in ("lying", "sitting"):
                return label

        rospy.loginfo("单帧姿态：三种朝向都检不到人 -> unknown")
        return "unknown"

    def appearance_match(self, result, tag="外观通道"):
        """用"外观"（ReID + 衣物颜色）判定身份，不要求人脸。"""
        return self.appearance_match_with(result, tag=tag, threshold=None)

    def appearance_match_with(self, result, tag="外观通道", threshold=None):
        """外观通道判定。threshold 为 None 时用 lying_match_threshold。"""
        score = float(result.get("score", -1.0))
        reid_score = float(result.get("reid_score", score))
        need = self.lying_match_threshold if threshold is None else float(threshold)
        matched = (
            score >= need
            and reid_score >= self.lying_min_reid_score
        )
        rospy.loginfo(
            "%s：score=%.3f(需≥%.2f) reid=%.3f(需≥%.2f) -> %s",
            tag,
            score,
            need,
            reid_score,
            self.lying_min_reid_score,
            "接受" if matched else "拒绝",
        )
        return matched

    def strict_channel_match(self, result, default_threshold):
        """第 1 步：常规判定（比父类更严格）。

        父类逻辑（owner_voice_reid_test.OwnerVoiceReidTest.result_is_match）
        在这两种情况下会退回"身形分数 >= match_threshold(0.7)"：
          a) 完全没检测到人脸（face_score is None）
          b) 人脸分数落在"既不算像、也不算不像"的中间地带(0.25~0.45)
        而身形+外观相似度对陌生人很容易到 0.7 以上（衣服颜色接近就行），
        结果陌生人被认成主人。

        这里改成：
          检测到人脸 -> 以脸为准：够像才接受，不够像直接拒绝
                        （身形分数不作为"翻盘"依据，因为实测陌生人的身形
                          分数能到 0.91，和真主人完全重叠，分不开）
          完全没有人脸 -> 只有身形分数 >= body_only_match_threshold 才接受
        """
        if result is None:
            return False

        score = float(result.get("score", -1.0))
        reid_score = float(result.get("reid_score", score))
        face_score = result.get("face_score")

        if face_score is not None:
            face_score = float(face_score)
            if face_score >= self.face_accept_threshold:
                return reid_score >= self.face_min_reid_score
            rospy.loginfo_throttle(
                2.0,
                "拒绝：检测到人脸但相似度 %.3f < %.2f（判定不是主人）",
                face_score,
                self.face_accept_threshold,
            )
            return False

        # 完全没有人脸时不要在这里放行，交给第 2 步按"姿态"决定：
        #   躺着 / 坐着（看不到脸是正常的）-> 走外观通道
        #   站着（典型就是背对相机的陌生人）-> 直接拒绝
        # 否则"背对相机且衣服颜色相近的陌生人"只要身形分数到 0.90
        # 就会被认成主人（实测确实会误认）。
        rospy.loginfo_throttle(
            2.0,
            "没有可用人脸（%s）-> 交由姿态判定决定，不再单凭身形放行",
            result.get("face_reason", ""),
        )
        return False

    # ------------------------------------------------------------------
    # 注册主流程：姓名 -> 确认 -> 人脸 -> 身形 -> 存盘
    # ------------------------------------------------------------------
    def record_all_owners(self):
        self.owner_profiles = []
        self.skipped_owner_indices = []
        for owner_index in range(1, self.owner_count + 1):
            if rospy.is_shutdown():
                return
            self.select_owner_profile_path(owner_index)
            # 重新注册这一位之前，先清掉他上一次的样本图（避免无限堆积，
            # 每位主人永远只有"最新一份"样本）
            self.reset_owner_sample_dirs(owner_index)
            # 换人前清空"当前这位"的中间状态（与基类保持一致）
            self.owner_name = ""
            self.owner_embedding = None
            self.owner_embedding_bank = None
            self.owner_color_embedding = None
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.current_record_face_embeddings = []
            self.owner_profile_meta = {}

            owner_name = self.wait_for_confirmed_owner_name(owner_index)
            if owner_name is None:
                # 说了"跳过"：这一位本轮不注册，同时把磁盘上原有的档案删掉，
                # 避免旧的主人信息留在那里被后续流程误用
                self.delete_owner_profile(owner_index)
                continue
            if not owner_name:
                return
            self.owner_name = owner_name

            rospy.loginfo(
                "Owner %d confirmed name: %s -> 开始采人脸", owner_index, owner_name
            )
            face_embeddings = self.capture_owner_face_only()

            rospy.loginfo(
                "Owner %d 人脸采集完成（%d 个样本）-> 开始采身形",
                owner_index,
                len(face_embeddings),
            )
            body_embeddings, body_color_embeddings, body_sample_meta = (
                self.capture_owner_body_only()
            )

            # 人脸 + 身形合并写入同一份档案：
            # save_owner_profile 会先把身形特征写进 npz，
            # 再读取 current_record_face_embeddings 把人脸补进同一份 npz。
            self.current_record_face_embeddings = face_embeddings
            self.save_owner_profile(
                body_embeddings,
                body_sample_meta,
                color_embeddings=body_color_embeddings,
            )
            self.remember_current_owner_profile()
            rospy.loginfo(
                "Owner %d profile saved: name=%s face=%d body=%d path=%s",
                owner_index,
                owner_name,
                len(face_embeddings),
                len(body_embeddings),
                self.profile_path,
            )
            self.speak("%s，注册完成。" % owner_name, wait=True)

        if not rospy.is_shutdown():
            self.face_ready = any(
                profile.get("face_embedding_bank") is not None
                for profile in self.owner_profiles
            )
            self.speak(self.all_profiles_recorded_text, wait=True)
            self.publish_status(
                "all_profiles_recorded",
                count=len(self.owner_profiles),
                registered_count=len(self.owner_profiles),
                skipped_count=len(self.skipped_owner_indices),
                skipped_owner_indices=list(self.skipped_owner_indices),
                face_ready=self.face_ready,
            )
            rospy.loginfo(
                "All owners recorded: %s",
                ", ".join(
                    "%d=%s" % (profile["index"], profile["name"])
                    for profile in self.owner_profiles
                ),
            )


def main():
    rospy.init_node("face_body_enroll_test")
    node = FaceBodyEnrollTest()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("face/body enroll test failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
