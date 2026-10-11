#!/usr/bin/env python3
# coding=utf-8

import json
import glob
import math
import os
import re
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET

import actionlib
import cv2
import numpy as np
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from person_reid_owner_test import PersonReidOwnerTest


class OwnerVoiceReidTest(PersonReidOwnerTest):
    def __init__(self):
        super().__init__()

        self.asr_topic = rospy.get_param("~asr_topic", "/voice/asr_text")
        self.answer_timeout = max(1.0, float(rospy.get_param("~answer_timeout", 12.0)))
        self.asr_wait_for_publishers = bool(rospy.get_param("~asr_wait_for_publishers", True))
        self.asr_wait_timeout = max(0.0, float(rospy.get_param("~asr_wait_timeout", 45.0)))
        self.max_name_retries = max(
            1,
            int(rospy.get_param("~max_name_retries", rospy.get_param("~max_retries", 3))),
        )
        self.max_name_length = max(1, int(rospy.get_param("~max_name_length", 12)))
        # 问名字后最多等多久；超时就重新问一遍，避免无声卡死
        self.name_answer_timeout = max(
            5.0, float(rospy.get_param("~name_answer_timeout", 30.0))
        )
        self.owner_count = max(1, int(rospy.get_param("~owner_count", 3)))
        self.owner_profiles = []
        self.skipped_owner_indices = []
        self.current_owner_index = 1
        self.last_match_owner_index = None
        self.base_profile_path = self.profile_path
        self.base_metadata_path = self.metadata_path

        self.face_verify_enabled = bool(rospy.get_param("~face_verify_enabled", True))
        self.face_auto_download = bool(rospy.get_param("~face_auto_download", False))
        self.face_model_name = rospy.get_param("~face_model_name", "buffalo_sc")
        self.face_model_root = os.path.expanduser(rospy.get_param("~face_model_root", "~/.insightface"))
        self.face_ctx_id = int(rospy.get_param("~face_ctx_id", -1))
        self.face_det_size = int(rospy.get_param("~face_det_size", 480))
        self.face_det_thresh = float(rospy.get_param("~face_det_thresh", 0.35))
        self.face_accept_threshold = float(rospy.get_param("~face_accept_threshold", 0.45))
        self.face_reject_threshold = float(rospy.get_param("~face_reject_threshold", 0.25))
        self.face_fast_reject = bool(rospy.get_param("~face_fast_reject", True))
        self.face_min_reid_score = float(rospy.get_param("~face_min_reid_score", 0.45))
        self.face_identity_weight = max(
            0.0,
            min(1.0, float(rospy.get_param("~face_identity_weight", 0.70))),
        )
        self.lying_face_identity_weight = max(
            0.0,
            min(1.0, float(rospy.get_param("~lying_face_identity_weight", 0.10))),
        )
        self.owner_score_margin_threshold = max(
            0.0,
            float(rospy.get_param("~owner_score_margin_threshold", 0.04)),
        )
        self.face_crop_padding = float(rospy.get_param("~face_crop_padding", 0.20))
        self.face_crop_top_ratio = float(rospy.get_param("~face_crop_top_ratio", 0.68))

        self.face_app = None
        self.face_model_ready = False
        self.face_ready = False
        self.owner_face_embedding = None
        self.owner_face_embedding_bank = None
        self.current_record_face_embeddings = []
        self.current_record_pose_id = ""
        self.current_record_pose_name = ""

        default_action_launch_file = os.path.abspath(
            os.path.join(
                SCRIPT_DIR,
                "..",
                "launch",
                "qwen_action_recognition.launch",
            )
        )
        self.action_recognition_enabled = bool(
            rospy.get_param("~action_recognition_enabled", True)
        )
        self.action_launch_file = os.path.abspath(
            os.path.expanduser(
                rospy.get_param("~action_launch_file", default_action_launch_file)
            )
        )
        self.action_node_name = str(
            rospy.get_param("~action_node_name", "owner_action_qwen")
        ).strip()
        self.action_result_topic = str(
            rospy.get_param(
                "~action_result_topic",
                "/owner_voice_reid_test/action_result",
            )
        ).strip()
        self.action_timeout = max(
            15.0,
            float(rospy.get_param("~action_timeout", 150.0)),
        )
        self.action_speech_grace = max(
            0.2,
            float(rospy.get_param("~action_speech_grace", 1.0)),
        )
        self.action_show_window = bool(
            rospy.get_param("~action_show_window", False)
        )
        # 动作识别节点自己也会播报结果（"主人正在躺着"）。总任务要播
        # "主人张三正在躺着"，所以那边会把它关掉；默认保持开启，老流程行为不变。
        self.action_speak_enabled = bool(
            rospy.get_param("~action_speak_enabled", True)
        )
        # ---- 动作节点复用 ----
        # 老做法：每位主人都重新拉起一个动作节点（启动+加载模型+预热+地面线
        # 标定+关闭，约 7 秒/人，其中标定和预热完全是重复劳动）。
        # 新做法：节点只启动一次（auto_analyze=false），每位主人用 trigger
        # 触发一次识别。判定算法本身完全没变，只是不再重复起节点。
        self.action_reuse_node = bool(
            rospy.get_param("~action_reuse_node", True)
        )
        self.action_trigger_topic = str(
            rospy.get_param(
                "~action_trigger_topic", "/pose_action_recognition/trigger"
            )
        ).strip()
        # 单次触发最多等多久（识别本身约 5.5 秒）。
        # 不要设太大：万一这次触发丢了，等的是这个时长。
        self.action_trigger_wait_seconds = max(
            3.0, float(rospy.get_param("~action_trigger_wait_seconds", 10.0))
        )
        # 等动作节点的 trigger 订阅者接上来（否则第一次触发会被静默丢掉）
        self.action_trigger_connect_timeout = max(
            1.0, float(rospy.get_param("~action_trigger_connect_timeout", 15.0))
        )
        # 等动作节点发"我已就绪"（模型加载 + 预热 + 地面线标定都完成）
        self.action_ready_topic = str(
            rospy.get_param(
                "~action_ready_topic", "/%s/ready" % self.action_node_name
            )
        ).strip()
        self.action_warmup_timeout = max(
            5.0, float(rospy.get_param("~action_warmup_timeout", 30.0))
        )
        self.action_ready_event = threading.Event()
        self.action_ready_sub = None
        # 节点还没预热好/正忙时的重试间隔
        self.action_trigger_retry_seconds = max(
            0.3, float(rospy.get_param("~action_trigger_retry_seconds", 1.5))
        )
        self.action_expected_request_id = ""
        self.action_trigger_pub = None
        self.action_request_seq = 0
        self.action_llm_url = str(
            rospy.get_param(
                "~action_llm_url",
                "http://127.0.0.1:11434/api/chat",
            )
        ).strip()
        self.action_llm_model = str(
            rospy.get_param("~action_llm_model", "qwen3.5:2b")
        ).strip()
        self.action_pointcloud_enabled = bool(
            rospy.get_param("~action_pointcloud_enabled", False)
        )
        self.action_startup_settle_seconds = max(
            0.0,
            float(rospy.get_param("~action_startup_settle_seconds", 0.6)),
        )
        self.action_capture_delay = max(
            0.0,
            float(rospy.get_param("~action_capture_delay", 0.3)),
        )
        self.action_capture_duration = max(
            0.0,
            float(rospy.get_param("~action_capture_duration", 5.0)),
        )
        self.action_capture_frame_count = max(
            1,
            int(rospy.get_param("~action_capture_frame_count", 9)),
        )
        self.action_llm_frame_count = max(
            1,
            int(rospy.get_param("~action_llm_frame_count", 5)),
        )
        self.action_pose_enabled = bool(
            rospy.get_param("~action_pose_enabled", True)
        )
        self.action_pose_model_path = str(
            rospy.get_param("~action_pose_model_path", "")
        ).strip()
        self.action_pose_device = str(
            rospy.get_param("~action_pose_device", "cpu")
        ).strip()
        self.action_pose_image_size = max(
            160,
            int(rospy.get_param("~action_pose_image_size", 416)),
        )
        self.action_pose_confidence = float(
            rospy.get_param("~action_pose_confidence", 0.25)
        )
        self.action_pose_iou = float(rospy.get_param("~action_pose_iou", 0.45))
        self.action_pose_max_detections = max(
            1,
            int(rospy.get_param("~action_pose_max_detections", 4)),
        )
        # 启动动作节点前先清掉上一轮残留的动作节点进程（任务被强杀时会留下）。
        self.action_cleanup_orphans = bool(
            rospy.get_param("~action_cleanup_orphans", True)
        )
        self.action_llm_timeout = max(
            1.0,
            float(rospy.get_param("~action_llm_timeout", 90.0)),
        )
        self.action_llm_keep_alive = str(
            rospy.get_param("~action_llm_keep_alive", "30m")
        ).strip()
        self.action_llm_max_tokens = max(
            1,
            int(rospy.get_param("~action_llm_max_tokens", 32)),
        )
        self.action_llm_num_ctx = max(
            512,
            int(rospy.get_param("~action_llm_num_ctx", 8192)),
        )
        self.action_jpeg_quality = max(
            1,
            min(100, int(rospy.get_param("~action_jpeg_quality", 65))),
        )
        self.action_image_max_width = max(
            0,
            int(rospy.get_param("~action_image_max_width", 384)),
        )
        self.action_model_warmup_retries = max(
            1,
            int(rospy.get_param("~action_model_warmup_retries", 3)),
        )
        self.action_model_warmup_retry_delay = max(
            0.0,
            float(rospy.get_param("~action_model_warmup_retry_delay", 2.0)),
        )
        self.action_use_owner_roi = bool(
            rospy.get_param("~action_use_owner_roi", True)
        )
        self.action_roi_padding = max(
            0.0,
            min(1.0, float(rospy.get_param("~action_roi_padding", 0.55))),
        )
        self.action_yolo_pause_settle_seconds = max(
            0.0,
            float(rospy.get_param("~action_yolo_pause_settle_seconds", 0.8)),
        )
        self.action_result_connect_timeout = max(
            0.0,
            float(rospy.get_param("~action_result_connect_timeout", 10.0)),
        )
        self.action_pause_yolo = bool(
            rospy.get_param("~action_pause_yolo", True)
        )
        self.yolo_pause_topic = str(
            rospy.get_param("~yolo_pause_topic", "/yoloworld/pause")
        ).strip()
        self.action_launch_parent = None
        self.action_process = None
        self.action_result_sub = None
        self.action_result_event = threading.Event()
        self.action_result = None
        self.action_launch_lock = threading.Lock()
        self.action_completed = False

        self.show_yolo_window = bool(rospy.get_param("~show_yolo_window", True))
        self.yolo_window_name = rospy.get_param("~yolo_window_name", "Owner YOLO-Person Box")
        self.yolo_window_ready = False
        self.yolo_window_failed = False

        self.navigate_enabled = bool(rospy.get_param("~navigate_enabled", True))
        self.waypoint_name = str(rospy.get_param("~waypoint_name", "living_room")).strip()
        self.waypoint_file = os.path.expanduser(
            rospy.get_param("~waypoint_file", "/home/dyp/waypoints.xml")
        )
        self.move_base_server_timeout = max(
            1.0,
            float(rospy.get_param("~move_base_server_timeout", 25.0)),
        )
        self.navigate_timeout = max(1.0, float(rospy.get_param("~navigate_timeout", 120.0)))
        self.clear_costmaps_before_navigation = bool(
            rospy.get_param("~clear_costmaps_before_navigation", True)
        )
        self.clear_costmaps_service = rospy.get_param(
            "~clear_costmaps_service", "/move_base/clear_costmaps"
        )
        self.clear_costmaps_timeout = max(
            0.1,
            float(rospy.get_param("~clear_costmaps_timeout", 5.0)),
        )
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.scan_angular_speed = float(rospy.get_param("~scan_angular_speed", -0.25))
        self.scan_total_angle = max(0.1, float(rospy.get_param("~scan_total_angle", math.pi)))
        self.scan_timeout = max(1.0, float(rospy.get_param("~scan_timeout", 35.0)))
        self.scan_step_angle = max(0.02, float(rospy.get_param("~scan_step_angle", 0.18)))
        self.scan_pause = max(0.0, float(rospy.get_param("~scan_pause", 0.35)))
        # 单次里程计读数之间允许的"真实转动"上限（弧度）。
        # 原代码是硬编码 0.7 rad（40°），太大：里程计噪声只要小于 40°
        # 就被当成真的在转，而且累加时取的是绝对值，噪声不抵消、全部
        # 累加成转动量 —— 实测只转了 30~50° 就被判成转满 180° 提前结束。
        self.scan_max_yaw_step = max(
            0.02, float(rospy.get_param("~scan_max_yaw_step", 0.25))
        )
        self.scan_after_arrival_delay = max(
            0.0,
            float(rospy.get_param("~scan_after_arrival_delay", 0.2)),
        )
        self.scan_return_to_owner = bool(rospy.get_param("~scan_return_to_owner", True))
        self.center_owner_enabled = bool(rospy.get_param("~center_owner_enabled", True))
        self.center_owner_timeout = max(0.5, float(rospy.get_param("~center_owner_timeout", 5.0)))
        self.center_owner_tolerance = max(
            0.01,
            float(rospy.get_param("~center_owner_tolerance", 0.06)),
        )
        self.center_owner_angular_gain = float(
            rospy.get_param("~center_owner_angular_gain", 0.65)
        )
        self.center_owner_max_angular_speed = abs(
            float(rospy.get_param("~center_owner_max_angular_speed", 0.30))
        )
        self.center_owner_lost_turn_speed = abs(
            float(rospy.get_param("~center_owner_lost_turn_speed", 0.10))
        )
        self.owner_not_found_text = rospy.get_param(
            "~owner_not_found_text", "没有找到主人。"
        )
        self.navigation_failed_text = rospy.get_param(
            "~navigation_failed_text", "导航到客厅失败。"
        )
        self.latest_yaw = None
        self.latest_odom_x = None
        self.latest_odom_y = None
        self.latest_odom_stamp = 0.0
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        self.name_prompt_text = rospy.get_param(
            "~name_prompt_text",
            "请说姓名",
        )
        self.indexed_name_prompt_text = rospy.get_param(
            "~indexed_name_prompt_text",
            "第%s位主人，请说「姓名」加您的名字，例如：姓名张三。",
        )
        self.name_failed_text = rospy.get_param(
            "~name_failed_text",
            "没有记录到主人姓名，测试结束。",
        )
        self.owner_skip_text = rospy.get_param(
            "~owner_skip_text",
            "已跳过第%s位主人的记录。",
        )
        self.first_owner_skip_text = rospy.get_param(
            "~first_owner_skip_text",
            "第一位主人不能跳过，请说姓名。",
        )
        self.asr_not_ready_text = rospy.get_param(
            "~asr_not_ready_text",
            "语音识别没有连接，请检查离线语音节点。",
        )
        self.name_recorded_text = rospy.get_param(
            "~name_recorded_text",
            "",
        )
        self.front_recording_text = rospy.get_param(
            "~front_recording_text",
            "请慢慢转向侧面。",
        )
        self.side_recording_text = rospy.get_param(
            "~side_recording_text",
            "请侧身面对相机，开始记录侧身特征。",
        )
        self.front_to_side_record_seconds = max(
            self.record_seconds,
            float(rospy.get_param("~front_to_side_record_seconds", 4.5)),
        )
        self.front_to_side_record_sample_count = max(
            self.record_min_samples,
            int(rospy.get_param("~front_to_side_record_sample_count", 10)),
        )
        self.front_to_side_record_sample_interval = max(
            0.05,
            float(rospy.get_param("~front_to_side_record_sample_interval", 0.35)),
        )
        self.owner_record_poses = [
            ("front_to_side", "正面转侧面", self.front_recording_text),
            ("side", "侧身", self.side_recording_text),
        ]
        self.all_profiles_recorded_text = rospy.get_param(
            "~all_profiles_recorded_text",
            "主人信息已全部记录完成",
        )
        self.named_owner_found_text = rospy.get_param(
            "~named_owner_found_text",
            "识别到主人，%s。",
        )

        self.owner_name = ""
        self.latest_name_answer = None
        self.accepting_name_answer = False
        self.name_condition = threading.Condition()

        self.result_pub = rospy.Publisher(
            rospy.get_param("~result_topic", "/owner_voice_reid_test/result"),
            String,
            queue_size=5,
            latch=True,
        )
        self.yolo_pause_pub = rospy.Publisher(
            self.yolo_pause_topic,
            Bool,
            queue_size=1,
            latch=True,
        )
        self.asr_sub = rospy.Subscriber(self.asr_topic, String, self.asr_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.on_shutdown(self.shutdown_action_recognition)
        rospy.on_shutdown(self.close_yolo_window)

    def wait_for_asr(self):
        if not self.asr_wait_for_publishers:
            return True
        deadline = time.time() + self.asr_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            try:
                if self.asr_sub.get_num_connections() > 0:
                    self.publish_status("asr_ready", topic=self.asr_topic)
                    return True
            except AttributeError:
                return True
            rospy.sleep(0.1)
        self.publish_status("asr_not_ready", topic=self.asr_topic, timeout=self.asr_wait_timeout)
        # 这一步在注册阶段是致命的（听不到主人姓名）。最常见的原因不是"节点崩了"，
        # 而是**起栈时没开 ASR**：bringup 里 start_asr 默认 false，只有 ⑦/⑦b 显式带了
        # start_asr:=true。所以日志里必须把"怎么修"写清楚，不然现场只能干看着。
        rospy.logerr(
            "No ASR publisher connected on %s（%.0f 秒内没有 /voice/asr_text 发布者）。"
            "最常见原因：起栈时没带 start_asr:=true —— bringup 里 start_asr 默认是 false！"
            "修法：① 用 VS Code 配置 ⑦/⑦b 重新起栈（它们显式带 start_asr:=true）；"
            "② 或不停栈，单独把离线语音挂上来："
            "roslaunch offline_voice_bridge offline_voice_zh.launch "
            "start_tts:=false start_sound_play:=false start_asr:=true "
            "asr_window_seconds:=2.5；③ 体检：bash tools/voice_health.sh",
            self.asr_topic,
            self.asr_wait_timeout,
        )
        self.speak(self.asr_not_ready_text, wait=True)
        return False

    def asr_callback(self, message):
        answer = message.data.strip()
        if not answer:
            return
        if self.is_likely_echo(answer):
            rospy.loginfo("ASR（自己的回声，已忽略）: %s", answer)
            return
        rospy.loginfo("ASR: %s", answer)
        with self.name_condition:
            if not self.accepting_name_answer:
                return
            self.latest_name_answer = answer
            self.name_condition.notify_all()

    def action_result_callback(self, message):
        try:
            result = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            rospy.logwarn("Invalid Qwen action result: %s", exc)
            return
        if not isinstance(result, dict):
            rospy.logwarn("Ignoring non-object Qwen action result")
            return
        # 复用模式：只收本次 trigger 的结果，晚到的旧结果直接丢掉
        expected = getattr(self, "action_expected_request_id", "")
        request_id = str(result.get("request_id", "") or "")
        if expected and request_id != expected:
            rospy.loginfo_throttle(
                2.0,
                "忽略不属于本次触发的动作结果：request_id=%s（本次=%s）",
                request_id or "(none)",
                expected,
            )
            return
        self.action_result = result
        self.action_result_event.set()

    @staticmethod
    def roslaunch_bool(value):
        return "true" if bool(value) else "false"

    @staticmethod
    def action_owner_roi(owner_result):
        candidate = owner_result.get("candidate", {}) if owner_result else {}
        bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
        if not bbox or len(bbox) < 4:
            return ""
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        except (TypeError, ValueError):
            return ""
        if x2 <= x1 or y2 <= y1:
            return ""
        return "%d,%d,%d,%d" % (x1, y1, x2, y2)

    def start_action_recognition(self, owner_result=None):
        """老路径：每次重新拉起一个动作节点，由它自己识别一次。"""
        self.shutdown_action_recognition()
        # 老路径的动作节点不带 request_id，这里清掉配对要求，否则结果会被当成"过期"
        self.action_expected_request_id = ""
        return self._launch_action_node(owner_result, auto_analyze=True)

    def start_action_node_once(self, owner_result=None):
        """新路径：动作节点只启动一次，之后靠 trigger 触发（已经在跑就复用）。

        节点用 auto_analyze=false 启动：它只做"加载模型 + 预热 + 地面线/相机
        高度标定"，不做识别；识别交给每一位主人的 trigger。
        这样标定只做一次，模型也只加载/预热一次。
        """
        with self.action_launch_lock:
            if self.action_launch_parent is not None:
                return True
        return self._launch_action_node(owner_result, auto_analyze=False)

    def cleanup_orphan_action_nodes(self, reason=""):
        """清掉上一轮遗留的动作节点进程（任务被强杀时它会活下来）。

        为什么要做：动作节点是 roslaunch 拉起来的子进程，父进程被 Ctrl-C 或被 VS Code
        强杀时它不会被回收，会一直占着 ~700MB 内存和一份模型权重。2026-10-09 实测：
        上午 11:44 启动的动作节点到下午 17:42 还活着（近 6 小时）。而且 ROS1 里新节点
        用同名注册会把旧的踢掉，留着只会互相干扰。

        只匹配本任务专用脚本名 ``qwen_action_recognition_node``，不碰相机/导航/语音。
        """
        me = os.getpid()
        known = set()
        child = getattr(self, "action_process", None)
        if getattr(child, "pid", None):
            known.add(int(child.pid))
        victims = []
        for proc_dir in glob.glob("/proc/[0-9]*"):
            try:
                pid = int(os.path.basename(proc_dir))
            except ValueError:
                continue
            if pid == me or pid in known:
                continue
            try:
                with open(os.path.join(proc_dir, "cmdline"), "rb") as handle:
                    cmdline = handle.read().decode("utf-8", "replace")
            except OSError:
                continue
            if "qwen_action_recognition_node" not in cmdline:
                continue
            victims.append(pid)
        if not victims:
            return
        rospy.logwarn(
            "%s：清掉 %d 个遗留的动作识别进程 %s（上一轮没被回收）",
            reason or "启动动作节点前",
            len(victims),
            victims,
        )
        for pid in victims:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if not any(os.path.exists("/proc/%d" % pid) for pid in victims):
                return
            time.sleep(0.2)
        for pid in victims:
            if os.path.exists("/proc/%d" % pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                    rospy.logwarn("遗留动作节点 pid %d 不吃 SIGTERM，已 SIGKILL", pid)
                except OSError:
                    pass

    def _launch_action_node(self, owner_result=None, auto_analyze=True):
        """把动作识别节点拉起来。

        auto_analyze=True ：节点起来后自己先识别一次（老路径：每位主人一个节点）
        auto_analyze=False：节点只做准备（加载模型/预热/标定），等 trigger
                            （新路径：节点常驻，每位主人触发一次）
        """
        if not self.action_recognition_enabled:
            self.publish_status("owner_action_disabled")
            return False
        if self.action_cleanup_orphans:
            self.cleanup_orphan_action_nodes()
        if not self.action_launch_file or not os.path.exists(self.action_launch_file):
            rospy.logwarn("Qwen action launch file not found: %s", self.action_launch_file)
            self.publish_status(
                "owner_action_unavailable",
                message="action launch file not found",
            )
            return False

        self.action_result = None
        self.action_result_event.clear()
        self.action_result_sub = rospy.Subscriber(
            self.action_result_topic,
            String,
            self.action_result_callback,
            queue_size=1,
        )

        try:
            import roslaunch

            launch_uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(launch_uuid)
            launch_args = [
                "start_camera:=false",
                "start_voice:=false",
                "start_sound_play:=false",
                "node_name:=%s" % self.action_node_name,
                "image_topic:=%s" % self.image_topic,
                "points_topic:=/kinect2/qhd/points",
                "say_topic:=%s" % self.say_topic,
                "result_topic:=%s" % self.action_result_topic,
                "trigger_topic:=%s" % self.action_trigger_topic,
                # 结论播报由谁来说：true=动作节点自己说，false=调用方说
                "speak_enabled:=%s" % self.roslaunch_bool(self.action_speak_enabled),
                # 被 trigger 触发时也要把结论念出来。
                # 老逻辑是节点自己开机识别一次（没有 request_id）才会播报；
                # 改成"触发式复用"以后，触发是带 request_id 的，不打开这个开关
                # 节点就一声不吭——实测 ③d/③e 会表现为"找到主人后卡住不动"。
                # ③f 那边把 speak_enabled 关掉了，所以这里打开也不会重复播报。
                "speak_on_trigger:=true",
                "show_window:=%s" % self.roslaunch_bool(self.action_show_window),
                "auto_analyze:=%s" % self.roslaunch_bool(auto_analyze),
                "auto_repeat_seconds:=0.0",
                "startup_settle_seconds:=%.3f" % self.action_startup_settle_seconds,
                "capture_delay:=%.3f" % self.action_capture_delay,
                "capture_duration:=%.3f" % self.action_capture_duration,
                "capture_frame_count:=%d" % self.action_capture_frame_count,
                "llm_frame_count:=%d" % self.action_llm_frame_count,
                "pose_enabled:=%s" % self.roslaunch_bool(self.action_pose_enabled),
                "pose_model_path:=%s" % self.action_pose_model_path,
                "pose_device:=%s" % self.action_pose_device,
                "pose_image_size:=%d" % self.action_pose_image_size,
                "pose_confidence:=%.3f" % self.action_pose_confidence,
                "pose_iou:=%.3f" % self.action_pose_iou,
                "pose_max_detections:=%d" % self.action_pose_max_detections,
                "llm_url:=%s" % self.action_llm_url,
                "llm_model:=%s" % self.action_llm_model,
                "llm_timeout:=%.3f" % self.action_llm_timeout,
                "llm_keep_alive:=%s" % self.action_llm_keep_alive,
                "llm_max_tokens:=%d" % self.action_llm_max_tokens,
                "llm_num_ctx:=%d" % self.action_llm_num_ctx,
                "jpeg_quality:=%d" % self.action_jpeg_quality,
                "image_max_width:=%d" % self.action_image_max_width,
                "model_warmup_retries:=%d" % self.action_model_warmup_retries,
                "model_warmup_retry_delay:=%.3f" % self.action_model_warmup_retry_delay,
                "pointcloud_enabled:=%s"
                % self.roslaunch_bool(self.action_pointcloud_enabled),
                "use_owner_roi:=%s" % self.roslaunch_bool(self.action_use_owner_roi),
                "owner_roi:=%s" % self.action_owner_roi(owner_result),
                "roi_padding:=%.3f" % self.action_roi_padding,
            ]
            launch_parent = roslaunch.parent.ROSLaunchParent(
                launch_uuid,
                [(self.action_launch_file, launch_args)],
            )
            if not auto_analyze:
                # 订阅"就绪"信号要在拉起节点之前，免得错过 latch 消息
                self.action_ready_event.clear()
                self.action_ready_sub = rospy.Subscriber(
                    self.action_ready_topic,
                    String,
                    self._action_ready_callback,
                    queue_size=1,
                )
            launch_parent.start()
            self.action_launch_parent = launch_parent
            connect_deadline = time.time() + self.action_result_connect_timeout
            while (
                not rospy.is_shutdown()
                and time.time() < connect_deadline
                and self.action_result_sub.get_num_connections() == 0
            ):
                rospy.sleep(0.05)
            if self.action_result_sub.get_num_connections() == 0:
                rospy.logwarn(
                    "Qwen action result topic has no publisher yet: %s",
                    self.action_result_topic,
                )
            if not auto_analyze:
                # 复用模式：等节点的 trigger 订阅者接上再返回。
                # 不然第一次触发会在"还没有订阅者"时被 rospy 静默丢掉，
                # 调用方只能干等一整轮超时（实测白等 21 秒）。
                if self.action_trigger_pub is None:
                    self.action_trigger_pub = rospy.Publisher(
                        self.action_trigger_topic, String, queue_size=1
                    )
                trigger_deadline = time.time() + self.action_trigger_connect_timeout
                while (
                    not rospy.is_shutdown()
                    and time.time() < trigger_deadline
                    and self.action_trigger_pub.get_num_connections() == 0
                ):
                    rospy.sleep(0.05)
                if self.action_trigger_pub.get_num_connections() == 0:
                    rospy.logwarn(
                        "动作触发话题仍没有订阅者：%s（第一次触发可能要重试一次）",
                        self.action_trigger_topic,
                    )
                else:
                    rospy.loginfo(
                        "动作节点已接上触发话题，可以开始触发：%s",
                        self.action_trigger_topic,
                    )
                # 再等节点把模型加载 / 预热 / 地面线标定做完（它完成时会发 ready）。
                # 不等的话前几次触发会被 "还没预热好" 拒掉，白等好几个 1.5 秒。
                ready_deadline = time.time() + self.action_warmup_timeout
                while (
                    not rospy.is_shutdown()
                    and time.time() < ready_deadline
                    and not self.action_ready_event.is_set()
                ):
                    rospy.sleep(0.05)
                if self.action_ready_event.is_set():
                    rospy.loginfo("动作节点已就绪（模型/预热/标定完成），开始触发")
                else:
                    rospy.logwarn(
                        "等动作节点就绪超时（%.0fs），仍按'重试'方式触发",
                        self.action_warmup_timeout,
                    )
            rospy.loginfo(
                "Started Qwen action node: name=%s result_topic=%s owner_roi=%s",
                self.action_node_name,
                self.action_result_topic,
                self.action_owner_roi(owner_result) or "disabled",
            )
            self.publish_status(
                "owner_action_started",
                node=self.action_node_name,
                result_topic=self.action_result_topic,
            )
            return True
        except Exception as exc:
            rospy.logwarn("Failed to start Qwen action node: %s", exc)
            self.publish_status("owner_action_unavailable", message=str(exc))
            self.shutdown_action_recognition()
            return False

    # ---------- 复用模式：常驻节点 + 逐次触发 ----------
    def _action_ready_callback(self, message):
        """动作节点报"我已就绪"（模型/预热/标定都完成）。"""
        self.action_ready_event.set()

    def trigger_action_node(self, owner_result=None):
        """给常驻的动作节点发一次触发请求，返回本次的 request_id。

        带上 request_id 是为了把"这次触发"和"这次的结果"配对：节点可能因为
        还没预热好或者正在忙而回一条错误结果，也可能上一轮的结果迟到，
        不配对就会张冠李戴。
        """
        if self.action_trigger_pub is None:
            self.action_trigger_pub = rospy.Publisher(
                self.action_trigger_topic, String, queue_size=1
            )
        self.action_request_seq += 1
        request_id = "owner-%s-%d" % (
            (owner_result or {}).get("owner_index", "x"),
            self.action_request_seq,
        )
        self.action_expected_request_id = request_id
        self.action_result = None
        self.action_result_event.clear()
        payload = {"request_id": request_id}
        # 只有调用方明确要求"只看主人 ROI"时才把 ROI 发过去。
        # 注意：节点把"触发里带的 ROI"当成最高优先级，会绕过它自己的
        # use_owner_roi 开关；之前无条件带 ROI，导致
        # _action_use_owner_roi:=false（想要全画面）被悄悄覆盖，
        # 裁到人身上之后姿态模型一个关键点都找不到（实测 pose=unknown(0.00)）。
        roi = self.action_owner_roi(owner_result) if self.action_use_owner_roi else ""
        if roi:
            payload["roi"] = roi
        self.action_trigger_pub.publish(String(data=json.dumps(payload)))
        rospy.loginfo(
            "Action trigger sent: request_id=%s roi=%s", request_id, roi or "disabled"
        )
        return request_id

    def wait_action_trigger_result(self, timeout):
        deadline = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.action_result_event.is_set():
                return self.action_result
            rospy.sleep(0.05)
        return None

    def run_action_recognition_reusing_node(self, owner_result):
        """常驻节点 + 逐次触发。

        节点还没预热完或者正忙时，会立刻回一条带 error 的结果，这里自动重试，
        不用调用方猜"要等多久才算准备好"。
        """
        deadline = time.time() + self.action_timeout
        attempt = 0
        while not rospy.is_shutdown() and time.time() < deadline:
            attempt += 1
            self.trigger_action_node(owner_result)
            result = self.wait_action_trigger_result(self.action_trigger_wait_seconds)
            if result is None:
                rospy.logwarn(
                    "动作节点这次触发没有回结果（第 %d 次），重试", attempt
                )
                continue
            if result.get("error"):
                rospy.loginfo(
                    "动作节点还没准备好或正忙（%s），%.1fs 后重试（第 %d 次）",
                    result.get("error"),
                    self.action_trigger_retry_seconds,
                    attempt,
                )
                rospy.sleep(self.action_trigger_retry_seconds)
                continue
            return result
        rospy.logwarn("动作识别（复用模式）在 %.0fs 内没有拿到有效结果", self.action_timeout)
        return None

    def set_yolo_paused(self, paused):
        if not self.action_pause_yolo or self.yolo_pause_pub is None:
            return False
        try:
            self.yolo_pause_pub.publish(Bool(data=bool(paused)))
            rospy.loginfo("YOLO-World %s for owner action recognition", "paused" if paused else "resumed")
            return True
        except Exception as exc:
            rospy.logwarn("Failed to set YOLO-World pause=%s: %s", paused, exc)
            return False

    def shutdown_action_recognition(self):
        with self.action_launch_lock:
            launch_parent = self.action_launch_parent
            self.action_launch_parent = None
            self.action_process = None
            result_sub = self.action_result_sub
            self.action_result_sub = None
        # 节点都关了，之前的 request_id 配对要求也一并清掉
        self.action_expected_request_id = ""

        if result_sub is not None:
            try:
                result_sub.unregister()
            except Exception:
                pass
        if launch_parent is not None:
            try:
                launch_parent.shutdown()
                rospy.loginfo("Stopped Qwen action node")
            except Exception as exc:
                rospy.logwarn("Failed to stop Qwen action node: %s", exc)

    def run_owner_action_recognition(self, owner_result):
        if self.action_completed:
            return self.action_result
        self.action_completed = True
        owner_name = owner_result.get("owner_name", self.owner_name) if owner_result else self.owner_name
        yolo_paused = False
        if self.action_pause_yolo:
            yolo_paused = self.set_yolo_paused(True)
            if yolo_paused and self.action_yolo_pause_settle_seconds > 0.0:
                rospy.loginfo(
                    "Waiting %.2fs for YOLO to settle before action recognition",
                    self.action_yolo_pause_settle_seconds,
                )
                rospy.sleep(self.action_yolo_pause_settle_seconds)
        try:
            if self.action_reuse_node:
                # 新路径：节点只启动一次，之后每位主人触发一次
                if not self.start_action_node_once(owner_result):
                    return None
                result = self.run_action_recognition_reusing_node(owner_result)
            else:
                # 老路径：每位主人重新拉起一个节点，由它自己识别一次
                if not self.start_action_recognition(owner_result):
                    return None
                deadline = time.time() + self.action_timeout
                rate = rospy.Rate(10)
                while not rospy.is_shutdown() and time.time() < deadline:
                    if self.action_result_event.is_set():
                        break
                    rate.sleep()
                result = self.action_result
            if result is None:
                rospy.logwarn(
                    "Timed out waiting for Qwen action result after %.1fs",
                    self.action_timeout,
                )
                self.publish_status(
                    "owner_action_timeout",
                    owner_name=owner_name,
                    timeout=self.action_timeout,
                )
                return None

            rospy.loginfo(
                "Owner action recognized: owner=%s action=%s place=%s",
                owner_name or "unknown",
                result.get("action", "unknown"),
                result.get("place", "unknown"),
            )
            self.publish_status(
                "owner_action_finished",
                owner_index=owner_result.get("owner_index") if owner_result else None,
                owner_name=owner_name,
                action=result.get("action", "unknown"),
                place=result.get("place", "unknown"),
                speech=result.get("speech", ""),
                recognizer=result.get("recognizer", "qwen"),
                latency_sec=result.get("total_sec"),
            )
            rospy.sleep(self.action_speech_grace)
            return result
        finally:
            # 复用模式下节点要保持活着给下一位主人用，收尾时统一关
            if not self.action_reuse_node:
                self.shutdown_action_recognition()
            if yolo_paused:
                self.set_yolo_paused(False)

    def odom_callback(self, message):
        orientation = message.pose.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self.lock:
            self.latest_yaw = yaw
            # 位置也要存：摔倒靠近的"最后一段爬行"是纯里程计闭环（点云在 <1 m
            # 看不到躺着的人），必须知道已经走了多远。
            self.latest_odom_x = float(message.pose.pose.position.x)
            self.latest_odom_y = float(message.pose.pose.position.y)
            self.latest_odom_stamp = time.time()

    def get_latest_odom_xy(self):
        """返回 (x, y, age)；没收到过里程计返回 (None, None, None)。"""
        with self.lock:
            if getattr(self, "latest_odom_x", None) is None:
                return None, None, None
            age = time.time() - float(getattr(self, "latest_odom_stamp", 0.0) or 0.0)
            return self.latest_odom_x, self.latest_odom_y, age

    def init_yolo_window(self):
        if not self.show_yolo_window or self.yolo_window_failed:
            return
        try:
            cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
            self.yolo_window_ready = True
            rospy.loginfo("YOLO debug window enabled: %s", self.yolo_window_name)
        except Exception as exc:
            self.yolo_window_ready = False
            self.yolo_window_failed = True
            rospy.logwarn("Unable to create YOLO debug window: %s", exc)

    def close_yolo_window(self):
        if not self.yolo_window_ready:
            return
        try:
            cv2.destroyWindow(self.yolo_window_name)
        except Exception:
            pass
        self.yolo_window_ready = False

    def update_yolo_window(self, status_text=""):
        if not self.yolo_window_ready:
            return
        with self.lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        for detection in detections:
            label = str(getattr(detection, "class_name", "person") or "person")
            score = float(getattr(detection, "score", 0.0) or 0.0)
            x1 = max(0, min(width - 1, int(getattr(detection, "xmin", 0))))
            y1 = max(0, min(height - 1, int(getattr(detection, "ymin", 0))))
            x2 = max(0, min(width - 1, int(getattr(detection, "xmax", 0))))
            y2 = max(0, min(height - 1, int(getattr(detection, "ymax", 0))))
            if x2 <= x1 or y2 <= y1:
                continue
            color = (0, 220, 0) if label.lower() == "person" else (0, 180, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = "%s %.2f" % (label, score)
            cv2.putText(
                image,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        header = "YOLO: %d persons" % sum(
            1
            for detection in detections
            if str(getattr(detection, "class_name", "person") or "person").lower() == "person"
        )
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
            rospy.logwarn_throttle(2.0, "YOLO debug window update failed: %s", exc)
            self.close_yolo_window()

    def get_latest_yaw(self):
        with self.lock:
            return self.latest_yaw

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def load_waypoint_pose(self, waypoint_name=None):
        target_name = str(waypoint_name or self.waypoint_name).strip()
        if not os.path.exists(self.waypoint_file):
            raise RuntimeError("waypoint file not found: %s" % self.waypoint_file)
        root = ET.parse(self.waypoint_file).getroot()
        for waypoint in root.findall("Waypoint"):
            if waypoint.findtext("Name", "").strip() != target_name:
                continue
            return {
                "x": float(waypoint.findtext("Pos_x", "0")),
                "y": float(waypoint.findtext("Pos_y", "0")),
                "z": float(waypoint.findtext("Ori_z", "0")),
                "w": float(waypoint.findtext("Ori_w", "1")),
            }
        raise RuntimeError("waypoint not found: %s in %s" % (target_name, self.waypoint_file))

    def clear_move_base_costmaps(self, reason):
        if not self.clear_costmaps_before_navigation:
            return
        try:
            rospy.wait_for_service(self.clear_costmaps_service, timeout=self.clear_costmaps_timeout)
            clear_costmaps = rospy.ServiceProxy(self.clear_costmaps_service, Empty)
            clear_costmaps()
            rospy.loginfo("Cleared move_base costmaps: %s", reason)
        except Exception as exc:
            rospy.logwarn("Unable to clear move_base costmaps %s: %s", reason, exc)

    def navigate_to_waypoint(self, waypoint_name=None):
        target_name = str(waypoint_name or self.waypoint_name).strip()
        if not self.navigate_enabled:
            rospy.loginfo("Navigation disabled; assuming robot is already at %s", target_name)
            return True

        rospy.loginfo("Waiting for move_base action server")
        if not self.move_base.wait_for_server(rospy.Duration(self.move_base_server_timeout)):
            raise RuntimeError("move_base action server is not available")

        pose = self.load_waypoint_pose(target_name)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = pose["x"]
        goal.target_pose.pose.position.y = pose["y"]
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.z = pose["z"]
        goal.target_pose.pose.orientation.w = pose["w"]

        self.clear_move_base_costmaps("before navigating to %s" % target_name)
        self.move_base.send_goal(goal)
        deadline = time.time() + self.navigate_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            self.update_yolo_window("导航到%s" % target_name)
            if self.move_base.wait_for_result(rospy.Duration(0.2)):
                break

        if rospy.is_shutdown():
            self.move_base.cancel_goal()
            self.stop_base()
            return False
        if not self.move_base.wait_for_result(rospy.Duration(0.0)):
            self.move_base.cancel_goal()
            self.stop_base()
            raise RuntimeError("move_base timed out while navigating to %s" % target_name)

        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            self.stop_base()
            raise RuntimeError("move_base failed with state %s" % state)

        self.stop_base()
        rospy.loginfo("Arrived at waypoint %s", target_name)
        return True

    @staticmethod
    def candidate_center_norm(candidate, image_width):
        bbox = candidate.get("bbox", [0, 0, image_width, 1])
        return ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(1.0, float(image_width))

    def center_owner_in_camera(self, owner_result):
        if not self.center_owner_enabled:
            return True

        initial_candidate = owner_result.get("candidate", {}) if owner_result else {}
        image, _detections = self.snapshot()
        if image is None:
            return False
        image_width = image.shape[1]
        last_center = self.candidate_center_norm(initial_candidate, image_width)
        latest_candidate = dict(initial_candidate)
        deadline = time.time() + self.center_owner_timeout
        centered_count = 0
        rate = rospy.Rate(12)

        while not rospy.is_shutdown() and time.time() < deadline:
            image, detections = self.snapshot()
            if image is None:
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if candidates:
                candidate = min(
                    candidates,
                    key=lambda item: abs(self.candidate_center_norm(item, image.shape[1]) - last_center),
                )
                latest_candidate = dict(candidate)
                last_center = self.candidate_center_norm(candidate, image.shape[1])
                x_error = last_center - 0.5
                if abs(x_error) <= self.center_owner_tolerance:
                    self.stop_base()
                    centered_count += 1
                    self.update_yolo_window("主人已居中")
                    if centered_count >= 3:
                        if owner_result is not None:
                            updated_candidate = dict(owner_result.get("candidate", {}))
                            updated_candidate.update(latest_candidate)
                            owner_result["candidate"] = updated_candidate
                        rospy.loginfo(
                            "Updated action ROI after centering: bbox=%s",
                            latest_candidate.get("bbox"),
                        )
                        rospy.loginfo("Owner centered in camera: center=%.3f", last_center)
                        return True
                else:
                    centered_count = 0
                    twist = Twist()
                    twist.angular.z = max(
                        -self.center_owner_max_angular_speed,
                        min(
                            self.center_owner_max_angular_speed,
                            -self.center_owner_angular_gain * x_error,
                        ),
                    )
                    self.cmd_pub.publish(twist)
                    self.update_yolo_window("主人居中中")
            else:
                centered_count = 0
                direction = 1.0 if last_center > 0.5 else -1.0
                twist = Twist()
                twist.angular.z = -direction * self.center_owner_lost_turn_speed
                self.cmd_pub.publish(twist)
                self.update_yolo_window("重新寻找主人")
            rate.sleep()

        self.stop_base()
        if owner_result is not None and latest_candidate:
            updated_candidate = dict(owner_result.get("candidate", {}))
            updated_candidate.update(latest_candidate)
            owner_result["candidate"] = updated_candidate
        rospy.logwarn("Owner centering timed out")
        return False

    def scan_for_owner(self):
        return self.scan_for_owner_for_target()

    def scan_for_owner_for_target(
        self, target_owner_index=None, not_found_text=None, followup=True,
        allowed_owner_indices=None, on_match=None,
    ):
        """原地旋转找主人。

        target_owner_index 为 None 时是原来的行为：找到"任意一位"已注册主人就返回。
        指定 owner_index 时只认那一位——总任务要"逐个找三位主人"，靠这个参数
        把每一轮锁定到具体某个人（其它主人从镜头前走过不会被误当成目标）。

        allowed_owner_indices：只认这个集合里的主人（"谁先出现先认谁"的模式用）。
        和 target_owner_index 二选一；两个都不传 = 任意一位都算。

        on_match：**一圈扫出多人**用。传了它就不是"认到第一位就停"，而是：
          · 每确认一位，就地调用 on_match(result)（由调用方做居中/播报/动作），
            然后继续转，把这一圈里剩下的主人都扫出来；
          · 已经确认过的那位从"还要认的人"里去掉，不会再被重复确认；
          · on_match 返回真值 = 请立刻结束这一圈（例如比赛时间快到了）；
          · 认完一圈后返回**已确认主人的列表**（一个都没认到就返回空列表）。
          为什么值得这么做：一个房间里可能同时站着两位主人，而"认到一位就收工"
          等于必须再转第二圈才能看到另一位 —— 那一圈就是纯浪费（实测一圈约 40 秒）。
          一位主人被确认的那一刻，他就在画面中间，所以"就地处理"不需要再转回去找他。
          on_match 为 None 时行为与以前完全一致。

        not_found_text：这一轮没找到时的播报。None = 用默认的
        ~owner_not_found_text；传空串则完全不播报（由调用方自己组织话术）。
        注意：用了 on_match（一圈扫多人）时这句**不会**播报 —— 扫完这一圈后
        是要接着去下一个房间、还是宣布"没找到"，得由调用方决定。

        followup：认到人之后要不要顺手做"居中 → 播报 → 动作识别"这一套。
        True（默认）= 老行为，③d 就靠它跑完全流程。
        False = 只把结果返回给调用方，由调用方自己决定接下来做什么。
        总任务（task_all_owners.py）必须传 False：它会自己做居中/播报/动作识别，
        否则一套动作识别会跑两遍，白等十几秒（实测踩过）。
        （on_match 和 followup 同时给时以 on_match 为准，避免跑两套处理。）
        """
        rospy.sleep(self.scan_after_arrival_delay)
        start_yaw = self.get_latest_yaw()
        deadline = time.time() + self.scan_timeout
        last_yaw = start_yaw
        rotated_angle = 0.0
        last_eval_time = 0.0
        last_match_owner_index = None
        match_consecutive_count = 0
        # 一圈扫多人模式：pending 是"这一圈还要认的主人"，认到一位就去掉一位
        collect_mode = on_match is not None
        pending = (
            set(allowed_owner_indices) if allowed_owner_indices is not None else None
        )
        collected = []
        stop_requested = False
        step_duration = self.scan_step_angle / max(abs(self.scan_angular_speed), 1e-3)
        moving = True
        phase_deadline = time.time() + step_duration
        rate = rospy.Rate(10)

        rospy.loginfo(
            "Scanning for owner: target=%s total_angle=%.2f rad step_angle=%.2f rad "
            "angular_speed=%.2f rad/s pause=%.2fs",
            (
                target_owner_index
                if target_owner_index is not None
                else (
                    sorted(allowed_owner_indices)
                    if allowed_owner_indices is not None
                    else "any"
                )
            ),
            self.scan_total_angle,
            self.scan_step_angle,
            self.scan_angular_speed,
            self.scan_pause,
        )
        while not rospy.is_shutdown() and time.time() < deadline:
            # 相机断流时不要继续转圈：识别的全是空帧，白烧时间预算，而且日志
            # 会看起来像"这间房没人"。父类只在这里调用（有定义的子类才生效），
            # 具体策略见 task_all_owners.hold_until_camera_recovers。
            holder = getattr(self, "hold_until_camera_recovers", None)
            if holder is not None:
                before_wait = time.time()
                if not holder("扫描房间"):
                    raise RuntimeError(
                        "相机在扫描房间过程中持续断流，看门狗没能救回来（"
                        "现场处理见 tools/camera_health.sh）"
                    )
                # 等相机的时间不算进"这一圈"的预算，否则一间房会白白少转一圈。
                deadline += time.time() - before_wait
            now = time.time()
            if moving:
                twist = Twist()
                twist.angular.z = self.scan_angular_speed
                self.cmd_pub.publish(twist)
                self.update_yolo_window("小步旋转寻找主人")
            else:
                self.stop_base()
                self.update_yolo_window("停顿观察画面")

            current_yaw = self.get_latest_yaw()
            if start_yaw is not None and current_yaw is not None and last_yaw is not None:
                delta = math.atan2(
                    math.sin(current_yaw - last_yaw),
                    math.cos(current_yaw - last_yaw),
                )
                if abs(delta) < self.scan_max_yaw_step:
                    # 带符号累加：里程计噪声正负对称，累加后自然抵消。
                    # 若写成 abs(delta)，噪声会被当成"真的转了"不断累加，
                    # 导致实际只转几十度就被判成转满一圈而提前结束。
                    rotated_angle += delta
                last_yaw = current_yaw

            rospy.loginfo_throttle(
                3.0,
                "寻找主人中：已累计转动 %.0f 度 / %.0f 度",
                math.degrees(abs(rotated_angle)),
                math.degrees(self.scan_total_angle),
            )

            now = time.time()
            if now - last_eval_time >= self.match_check_interval:
                result = self.evaluate_current_frame()
                last_eval_time = now
                if result is None:
                    # 这一轮连候选都没有（画面里没人，或检测被跳过）。
                    # 以前这里是静默的，害得"连续命中计数"被悄悄扣掉却查不出原因。
                    rospy.loginfo_throttle(
                        3.0, "本轮没有可用候选（画面里没人或检测被跳过）"
                    )
                matched = (
                    result is not None
                    and self.result_is_match(result, self.match_threshold)
                    and (
                        (
                            target_owner_index is None
                            and pending is None
                        )
                        or (
                            target_owner_index is not None
                            and result.get("owner_index") == target_owner_index
                        )
                        or (
                            pending is not None
                            and result.get("owner_index") in pending
                        )
                    )
                )
                if matched:
                    owner_index = result.get("owner_index")
                    required_consecutive = result.get(
                        "required_consecutive",
                        self.match_required_consecutive,
                    )
                    if last_match_owner_index == owner_index:
                        match_consecutive_count += 1
                    else:
                        match_consecutive_count = 1
                    last_match_owner_index = owner_index
                    self.publish_status(
                        "scan_match_score",
                        owner_index=owner_index,
                        name=result.get("owner_name", ""),
                        score=result.get("score"),
                        identity_score=result.get("identity_score"),
                        reid_score=result.get("reid_score"),
                        face_score=result.get("face_score"),
                        owner_score_margin=result.get("owner_score_margin"),
                        consecutive=match_consecutive_count,
                        required_consecutive=required_consecutive,
                    )
                    if match_consecutive_count >= required_consecutive:
                        self.stop_base()
                        rospy.sleep(0.2)
                        if not collect_mode:
                            if followup:
                                self.center_owner_in_camera(result)
                                self.announce_owner_result(result)
                                self.run_owner_action_recognition(result)
                            return result
                        # ---- 一圈扫多人：就地处理这一位，然后接着转 ----
                        # 他被确认的这一刻正在画面中间，所以居中/动作识别当场就能做，
                        # 不需要再转回去找他。
                        collected.append(result)
                        if pending is not None:
                            pending.discard(owner_index)
                        rospy.loginfo(
                            "这一圈确认第 %d 位：%s；这一圈还要认：%s",
                            len(collected),
                            result.get("owner_name", owner_index),
                            "、".join(str(i) for i in sorted(pending))
                            if pending
                            else "（没有了）",
                        )
                        handle_started = time.time()
                        handle_yaw_before = self.get_latest_yaw()
                        try:
                            stop_requested = bool(on_match(result))
                        except Exception as exc:
                            rospy.logwarn("on_match 处理这位主人时出错：%s", exc)
                        # 处理期间花掉的时间和转动不算进"这一圈"：
                        # 居中时会小幅转动，若算进去，圈还没转完就会被判成转满、
                        # 提前结束扫描，把后面的人漏掉。
                        deadline += max(0.0, time.time() - handle_started)
                        handle_yaw_after = self.get_latest_yaw()
                        if (
                            handle_yaw_before is not None
                            and handle_yaw_after is not None
                        ):
                            handle_delta = math.atan2(
                                math.sin(handle_yaw_after - handle_yaw_before),
                                math.cos(handle_yaw_after - handle_yaw_before),
                            )
                            if abs(handle_delta) < math.pi:
                                rotated_angle -= handle_delta
                        last_yaw = handle_yaw_after
                        match_consecutive_count = 0
                        last_match_owner_index = None
                        if stop_requested:
                            rospy.loginfo("收到「结束这一圈」的请求，提前停止扫描")
                            break
                else:
                    # 不做"一次失败就清零"：躺姿的姿态判定本身会抖动
                    # （同一个姿势可能这次判 lying、下次判 sitting），
                    # 而且 ReID 分数也在阈值附近上下浮动。清零会导致
                    # 永远凑不满"连续 N 次"，主人再怎么躺也找不到。
                    # 改成每次失败减一：偶发失败不会把进度全部抹掉，
                    # 连续失败仍然会迅速归零（保护陌生人场景）。
                    if match_consecutive_count > 0:
                        match_consecutive_count -= 1
                    if match_consecutive_count <= 0:
                        last_match_owner_index = None

            if start_yaw is not None and abs(rotated_angle) >= self.scan_total_angle:
                break

            now = time.time()
            if now >= phase_deadline:
                if moving:
                    self.stop_base()
                    moving = False
                    phase_deadline = now + self.scan_pause
                else:
                    moving = True
                    phase_deadline = now + step_duration
            rate.sleep()

        self.stop_base()
        if start_yaw is not None:
            rospy.loginfo("Owner scan completed: rotated %.2f rad", rotated_angle)
        else:
            rospy.logwarn("No odom yaw received on %s; scan ended by timeout", self.odom_topic)
        if collect_mode:
            rospy.loginfo(
                "这一圈共确认 %d 位主人：%s",
                len(collected),
                "、".join(str(item.get("owner_name", "")) for item in collected)
                or "无",
            )
            self.publish_status(
                "owner_scan_collected",
                rotated_angle=rotated_angle,
                collected=[item.get("owner_index") for item in collected],
                stop_requested=stop_requested,
            )
            return collected
        final_not_found_text = self.owner_not_found_text if not_found_text is None else not_found_text
        if final_not_found_text:
            self.speak(final_not_found_text, wait=True)
        self.publish_status(
            "owner_not_found",
            rotated_angle=rotated_angle,
            target_owner_index=target_owner_index,
        )
        return None

    @staticmethod
    def clean_text(text):
        return re.sub(r"\s+", "", text.strip().strip("，。！？、,.!?;；:： "))

    @staticmethod
    def clamp_value(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6 or not np.isfinite(norm):
            return None
        return vector / norm

    @staticmethod
    def select_largest_face(faces):
        if not faces:
            return None

        def face_priority(face):
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return 0.0
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            det_score = float(getattr(face, "det_score", 1.0) or 1.0)
            return width * height * det_score

        return max(faces, key=face_priority)

    def init_face_recognizer(self):
        if not self.face_verify_enabled:
            rospy.loginfo("InsightFace fusion disabled")
            return

        model_dir = os.path.join(self.face_model_root, "models", self.face_model_name)
        cached_models = []
        if os.path.isdir(model_dir):
            cached_models = [name for name in os.listdir(model_dir) if name.endswith(".onnx")]
        if not cached_models and not self.face_auto_download:
            rospy.logwarn(
                "InsightFace model %s is not cached under %s. Face fusion will be disabled.",
                self.face_model_name,
                model_dir,
            )
            return

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            rospy.logwarn("InsightFace is not available: %s. Face fusion will be disabled.", exc)
            return

        # 人脸推理设备：~face_ctx_id < 0 → CPU；>= 0 → CUDA（第 ctx_id 号卡）。
        #
        # 注意（2026-10-09 实测踩坑）：这里**必须显式写 provider**。
        # onnxruntime 的 InferenceSession(providers=None) 不是"自动选最快的"，
        # 而是"不指定执行提供者"= 只用 CPUExecutionProvider（源码
        # onnxruntime_inference_collection.py: providers is None -> return [], []）。
        # 之前的写法在 ctx_id>=0 时传 None，看着像开了 GPU，实际日志里是
        # "Applied providers: ['CPUExecutionProvider']" —— 人脸一直在 CPU 上跑。
        # 显式给出 ['CUDAExecutionProvider', 'CPUExecutionProvider'] 后：CUDA 可用就用
        # GPU，装不上/加载失败时 ORT 会自己降级到列表里的 CPU，不会让注册流程挂掉。
        if self.face_ctx_id < 0:
            providers = ["CPUExecutionProvider"]
        else:
            providers = [
                ("CUDAExecutionProvider", {"device_id": self.face_ctx_id}),
                "CPUExecutionProvider",
            ]
        try:
            self.face_app = FaceAnalysis(
                name=self.face_model_name,
                root=self.face_model_root,
                providers=providers,
            )
            self.face_app.prepare(
                ctx_id=self.face_ctx_id,
                det_thresh=self.face_det_thresh,
                det_size=(self.face_det_size, self.face_det_size),
            )
            self.face_model_ready = True
            rospy.loginfo(
                "InsightFace fusion ready: model=%s ctx_id=%d providers=%s "
                "det_size=%d threshold=%.2f",
                self.face_model_name,
                self.face_ctx_id,
                providers,
                self.face_det_size,
                self.face_accept_threshold,
            )
        except Exception as exc:
            self.face_app = None
            self.face_model_ready = False
            rospy.logwarn("Failed to initialize InsightFace fusion: %s", exc)

    def candidate_face_regions(self, crop, candidate=None):
        if crop is None or crop.size == 0:
            return []
        height, width = crop.shape[:2]
        top_ratio = self.clamp_value(float(self.face_crop_top_ratio), 0.35, 1.0)
        padding = self.clamp_value(float(self.face_crop_padding), 0.0, 0.40)

        upper_y2 = max(1, int(height * top_ratio))
        pad_x = int(width * padding)
        x1 = int(self.clamp_value(0 - pad_x, 0, width - 1))
        x2 = int(self.clamp_value(width + pad_x, x1 + 1, width))
        regions = [
            ("upper", crop[0:upper_y2, x1:x2]),
            ("full", crop),
        ]
        return [(name, image) for name, image in regions if image is not None and image.size > 0]

    def extract_face_embedding(self, crop, candidate=None):
        if not self.face_model_ready or self.face_app is None:
            return None, 0.0, "face unavailable"

        best_embedding = None
        best_area = 0.0
        best_min_side = 0.0
        best_region = ""
        face_count = 0
        elapsed_ms = 0.0
        for region_name, face_image in self.candidate_face_regions(crop, candidate=candidate):
            try:
                start = time.time()
                faces = self.face_app.get(face_image)
                elapsed_ms += (time.time() - start) * 1000.0
            except Exception as exc:
                rospy.logwarn("InsightFace failed on %s crop: %s", region_name, exc)
                continue

            face = self.select_largest_face(faces)
            if face is None:
                continue
            face_count += len(faces) if faces is not None else 0
            embedding = self.normalize_embedding(face.embedding)
            if embedding is None:
                continue
            bbox = getattr(face, "bbox", None)
            if bbox is not None and len(bbox) >= 4:
                face_w = max(0.0, float(bbox[2]) - float(bbox[0]))
                face_h = max(0.0, float(bbox[3]) - float(bbox[1]))
                area = face_w * face_h
            else:
                area = 1.0
                face_w = face_h = 0.0
            if area > best_area:
                best_embedding = embedding
                best_area = area
                best_min_side = min(face_w, face_h)
                best_region = region_name
            if best_embedding is not None:
                break

        if best_embedding is None:
            return None, elapsed_ms, "no face"
        return {
            "embedding": best_embedding,
            "region": best_region,
            "face_count": face_count,
            # 人脸框在裁剪图里的短边像素数。人脸越小，特征越不可信 ——
            # 2~3 米外的人脸只有三四十像素，同一段视频里相似度能在 0.09~0.44
            # 之间乱跳（实测），这种"人脸"不该有资格一票否决。
            "min_side": best_min_side,
        }, elapsed_ms, "face ready"

    def parse_owner_name(self, answer):
        text = self.clean_text(answer)
        # 上游只认「姓名张三 / 姓名是张三」这种说法，现场直接报「张三」
        # 会被判成空名字，然后一直静静地等下去（实测卡死就卡在这里）。
        # 所以这里改成：先按上游写法解析，解析不到再退回到自然说法。
        match = re.match(r"^(?:主人)?姓名(?:是|叫|为|:|：)?(.+)$", text)
        if match:
            name = match.group(1)
        else:
            prefix_match = re.search(
                r"(?:我的名字叫|我的名字是|名字叫|名字是|我叫|我是)"
                r"([一-龥A-Za-z][一-龥A-Za-z0-9·_-]{0,11})",
                text,
            )
            name = prefix_match.group(1) if prefix_match else text
        name = name.strip("，。！？、,.!?;；:： ")
        name = re.split(r"[，。！？、,.!?;；:：]", name, maxsplit=1)[0]
        name = re.sub(r"^(?:是|叫|为)", "", name).strip()
        name = self.clean_text(name)
        name = re.sub(r"[^一-龥A-Za-z0-9·_-]", "", name)
        if not name:
            return ""
        if any(
            marker in name
            for marker in ("名字", "姓名", "什么", "叫我", "说", "回答", "吗", "呢")
        ):
            return ""
        if name in (
            "重来",
            "重新",
            "重说",
            "再说",
            "没有",
            "没有听清",
            "听不清",
            "不是",
            "不对",
            "确认",
            "正确",
            "是的",
            "好的",
            "确定",
        ):
            return ""
        if len(name) < 2:
            return ""
        return name[: self.max_name_length]

    def is_skip_owner_answer(self, answer):
        text = self.clean_text(answer)
        if not text:
            return False
        if text in ("跳过", "跳過", "略过", "略過", "下一个", "下一位"):
            return True
        # 语音识别常把“跳过”带上语气词或尾字（跳过吧 / 跳过这位），这里放宽一点
        return any(token in text for token in ("跳过", "跳過", "略过", "略過"))

    @staticmethod
    def format_with_name(template, name):
        if "%s" in template:
            return template % name
        if "{name}" in template:
            return template.format(name=name)
        return "%s%s" % (template, name)

    @staticmethod
    def owner_index_label(owner_index):
        labels = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        if 1 <= owner_index <= len(labels):
            return labels[owner_index - 1]
        return str(owner_index)

    def format_owner_index_text(self, template, owner_index):
        label = self.owner_index_label(owner_index)
        if "%s" in template:
            return template % label
        if "%d" in template:
            return template % owner_index
        if "{index}" in template or "{label}" in template:
            return template.format(index=owner_index, label=label)
        return template

    def owner_profile_paths(self, owner_index):
        profile_root, profile_ext = os.path.splitext(self.base_profile_path)
        metadata_root, metadata_ext = os.path.splitext(self.base_metadata_path)
        if self.owner_count == 1:
            return self.base_profile_path, self.base_metadata_path
        return (
            "%s_%d%s" % (profile_root, owner_index, profile_ext or ".npz"),
            "%s_%d%s" % (metadata_root, owner_index, metadata_ext or ".json"),
        )

    def select_owner_profile_path(self, owner_index):
        self.current_owner_index = owner_index
        self.profile_path, self.metadata_path = self.owner_profile_paths(owner_index)

    def wait_for_owner_name(self, owner_index=None, prompt_text=None):
        """问姓名并等回答。

        prompt_text 不为空时用它替代标准问句。重试时调用方会传一句短话
        （例如"好的，请再说一次您的名字。"），这样"致歉 + 重新提问"合成
        一句话，省掉原来两句播报的 4 秒左右。
        """
        with self.name_condition:
            self.latest_name_answer = None
            self.accepting_name_answer = False

        if prompt_text is None:
            if owner_index is None:
                prompt_text = self.name_prompt_text
            else:
                prompt_text = self.format_owner_index_text(
                    self.indexed_name_prompt_text, owner_index
                )
        self.speak(prompt_text, wait=True)

        with self.name_condition:
            self.latest_name_answer = None
            self.accepting_name_answer = True
            while not rospy.is_shutdown():
                deadline = time.time() + self.name_answer_timeout
                while (
                    self.latest_name_answer is None
                    and not rospy.is_shutdown()
                    and time.time() < deadline
                ):
                    self.name_condition.wait(timeout=0.2)
                    self.update_yolo_window("等待第%s位主人姓名" % self.owner_index_label(self.current_owner_index))
                if self.latest_name_answer is None:
                    # 没人回答（或者回答没被解析出来）时重新问一遍，
                    # 否则节点会一声不响地干等，看不出是死机还是在等。
                    rospy.logwarn(
                        "等待姓名超过 %.0f 秒仍没有可用回答，重新提示",
                        self.name_answer_timeout,
                    )
                    self.publish_status(
                        "owner_name_timeout", owner_index=self.current_owner_index
                    )
                    self.speak(prompt_text, wait=True)
                    continue
                raw_answer = self.latest_name_answer or ""
                self.latest_name_answer = None
                if not raw_answer:
                    continue

                if self.is_skip_owner_answer(raw_answer):
                    if owner_index is None or owner_index <= 1:
                        self.publish_status(
                            "owner_skip_ignored",
                            owner_index=self.current_owner_index,
                            raw=raw_answer,
                            reason="first_owner_required",
                        )
                        self.speak(self.first_owner_skip_text, wait=True)
                        continue

                    self.accepting_name_answer = False
                    self.skipped_owner_indices.append(self.current_owner_index)
                    payload = {
                        "event": "owner_skipped",
                        "owner_index": self.current_owner_index,
                        "raw": raw_answer,
                    }
                    self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                    self.publish_status(
                        "owner_skipped",
                        owner_index=self.current_owner_index,
                        raw=raw_answer,
                    )
                    self.speak(
                        self.format_owner_index_text(self.owner_skip_text, self.current_owner_index),
                        wait=True,
                    )
                    rospy.loginfo("Owner %d enrollment skipped by voice command", self.current_owner_index)
                    return None

                name = self.parse_owner_name(raw_answer)
                if not name:
                    rospy.logwarn("没能从这句话里听出姓名，忽略：%s", raw_answer)
                    self.publish_status("owner_name_ignored", raw=raw_answer)
                    continue

                self.owner_name = name
                payload = {
                    "event": "owner_name_recorded",
                    "owner_index": self.current_owner_index,
                    "name": name,
                    "raw": raw_answer,
                }
                self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                self.publish_status(
                    "owner_name_recorded",
                    owner_index=self.current_owner_index,
                    name=name,
                    raw=raw_answer,
                )
                self.accepting_name_answer = False
                # 这段"已记录主人姓名"属于多余复述，默认文案为空；
                # 留了参数口子，需要时可用 ~name_recorded_text 覆盖回来。
                if str(self.name_recorded_text or "").strip():
                    self.speak(self.format_with_name(self.name_recorded_text, name), wait=True)
                rospy.loginfo("Owner %d name recorded: %s raw=%s", self.current_owner_index, name, raw_answer)
                return name
            self.accepting_name_answer = False
        self.speak(self.name_failed_text, wait=True)
        return ""

    def save_crop(self, crop, directory, prefix, index):
        path = super().save_crop(crop, directory, prefix, index)
        if prefix.startswith("owner") and self.face_model_ready:
            face_data, elapsed_ms, reason = self.extract_face_embedding(crop)
            if face_data is None:
                rospy.logwarn(
                    "No usable face embedding for owner %d %s sample %d: %s elapsed=%.1fms",
                    self.current_owner_index,
                    self.current_record_pose_name or prefix,
                    index,
                    reason,
                    elapsed_ms,
                )
                return path
            self.current_record_face_embeddings.append(
                {
                    "sample": index,
                    "pose": self.current_record_pose_id,
                    "pose_name": self.current_record_pose_name,
                    "image": path,
                    "region": face_data.get("region", ""),
                    "face_count": face_data.get("face_count", 0),
                    "embedding": face_data["embedding"],
                }
            )
            self.publish_status(
                "face_recording_sample",
                owner_index=self.current_owner_index,
                sample=index,
                pose=self.current_record_pose_id,
                pose_name=self.current_record_pose_name,
                region=face_data.get("region", ""),
                faces=face_data.get("face_count", 0),
                elapsed_ms=elapsed_ms,
            )
        return path

    def record_owner_pose(
        self,
        pose_id,
        pose_name,
        prompt_text,
        sample_offset,
        duration=None,
        sample_count=None,
        sample_interval=None,
    ):
        self.current_record_pose_id = pose_id
        self.current_record_pose_name = pose_name
        self.publish_status(
            "recording_pose_started",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            pose=pose_id,
            pose_name=pose_name,
        )
        self.speak(prompt_text, wait=True)
        self.play_ding()

        embeddings = []
        color_embeddings = []
        sample_meta = []
        sample_dir = self.profile_sample_dir()
        capture_duration = self.record_seconds if duration is None else max(0.5, float(duration))
        target_sample_count = self.record_sample_count if sample_count is None else max(1, int(sample_count))
        capture_interval = (
            self.record_sample_interval
            if sample_interval is None
            else max(0.05, float(sample_interval))
        )
        deadline = time.time() + capture_duration
        next_sample_time = 0.0
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and time.time() < deadline and len(embeddings) < target_sample_count:
            now = time.time()
            if now < next_sample_time:
                self.update_yolo_window("登记%s" % pose_name)
                rate.sleep()
                continue
            next_sample_time = now + capture_interval
            image, detections = self.snapshot()
            if image is None:
                rospy.logwarn_throttle(1.0, "Waiting for fresh camera image on %s", self.image_topic)
                self.update_yolo_window("等待相机")
                rate.sleep()
                continue
            candidates = self.person_candidates(image, detections)
            if not candidates:
                rospy.logwarn_throttle(1.0, "Waiting for person detection on %s", self.detections_topic)
                self.update_yolo_window("等待YOLO人体框")
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
            pose_sample_index = len(embeddings) + 1
            sample_index = sample_offset + pose_sample_index
            path = self.save_crop(crop, sample_dir, "owner_%s" % pose_id, sample_index)
            color_embedding = self.color_hist_embedding(crop)
            if color_embedding is not None:
                color_embeddings.append(color_embedding)
            embeddings.append(extracted[0])
            sample_meta.append(
                {
                    "index": sample_index,
                    "pose": pose_id,
                    "pose_name": pose_name,
                    "pose_sample": pose_sample_index,
                    "image": path,
                    "bbox": candidates[0]["bbox"],
                    "crop_bbox": crop_bbox,
                    "det_score": candidates[0]["score"],
                    "area_ratio": candidates[0]["area_ratio"],
                }
            )
            self.publish_status(
                "recording_sample",
                owner_index=self.current_owner_index,
                sample=sample_index,
                pose=pose_id,
                pose_name=pose_name,
                pose_sample=pose_sample_index,
                required=self.record_min_samples,
                target=target_sample_count,
            )
            rospy.loginfo(
                "Owner %d %s Re-ID sample %d captured: %s",
                self.current_owner_index,
                pose_name,
                pose_sample_index,
                path or "not saved",
            )
            self.update_yolo_window("登记%s" % pose_name)
            rate.sleep()

        self.current_record_pose_id = ""
        self.current_record_pose_name = ""
        if len(embeddings) < self.record_min_samples:
            self.publish_status(
                "recording_pose_failed",
                owner_index=self.current_owner_index,
                pose=pose_id,
                pose_name=pose_name,
                samples=len(embeddings),
                required=self.record_min_samples,
            )
            self.speak(self.record_failed_text, wait=True)
            raise RuntimeError(
                "only captured %d/%d usable %s Re-ID samples"
                % (len(embeddings), self.record_min_samples, pose_name)
            )

        self.publish_status(
            "recording_pose_done",
            owner_index=self.current_owner_index,
            pose=pose_id,
            pose_name=pose_name,
            samples=len(embeddings),
        )
        return embeddings, color_embeddings, sample_meta

    def record_owner(self):
        self.publish_status(
            "recording_started",
            owner_index=self.current_owner_index,
            name=self.owner_name,
            poses=[pose_name for _, pose_name, _ in self.owner_record_poses],
        )
        all_embeddings = []
        all_color_embeddings = []
        all_sample_meta = []
        for pose_id, pose_name, prompt_text in self.owner_record_poses:
            duration = None
            sample_count = None
            sample_interval = None
            if pose_id == "front_to_side":
                duration = self.front_to_side_record_seconds
                sample_count = self.front_to_side_record_sample_count
                sample_interval = self.front_to_side_record_sample_interval
            embeddings, color_embeddings, sample_meta = self.record_owner_pose(
                pose_id,
                pose_name,
                prompt_text,
                len(all_embeddings),
                duration=duration,
                sample_count=sample_count,
                sample_interval=sample_interval,
            )
            all_embeddings.extend(embeddings)
            all_color_embeddings.extend(color_embeddings)
            all_sample_meta.extend(sample_meta)

        self.save_owner_profile(
            all_embeddings,
            all_sample_meta,
            color_embeddings=all_color_embeddings,
        )
        self.speak(self.record_done_text, wait=True)
        self.publish_status(
            "recording_done",
            owner_index=self.current_owner_index,
            samples=len(all_embeddings),
            poses=[pose_id for pose_id, _, _ in self.owner_record_poses],
        )

    def remember_current_owner_profile(self):
        profile = {
            "index": self.current_owner_index,
            "name": self.owner_name,
            "embedding": self.owner_embedding,
            "embedding_bank": self.owner_embedding_bank,
            "color_embedding": self.owner_color_embedding,
            "face_embedding": self.owner_face_embedding,
            "face_embedding_bank": self.owner_face_embedding_bank,
            "metadata": dict(self.owner_profile_meta),
            "profile_path": self.profile_path,
            "metadata_path": self.metadata_path,
        }
        existing_index = next(
            (index for index, item in enumerate(self.owner_profiles) if item["index"] == self.current_owner_index),
            None,
        )
        if existing_index is None:
            self.owner_profiles.append(profile)
        else:
            self.owner_profiles[existing_index] = profile
        self.owner_profiles.sort(key=lambda item: item["index"])

    def record_all_owners(self):
        self.owner_profiles = []
        self.skipped_owner_indices = []
        for owner_index in range(1, self.owner_count + 1):
            if rospy.is_shutdown():
                return
            self.select_owner_profile_path(owner_index)
            # 重新注册这一位之前，先清掉他上一次的样本图，避免越攒越多
            self.reset_owner_sample_dirs(owner_index)
            self.owner_name = ""
            self.owner_embedding = None
            self.owner_embedding_bank = None
            self.owner_color_embedding = None
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.current_record_face_embeddings = []
            self.owner_profile_meta = {}
            owner_name = self.wait_for_owner_name(owner_index)
            if owner_name is None:
                # 说了"跳过"：删掉这一位原有的档案，保证磁盘上的档案集合
                # 就是本轮真正注册成功的人
                self.delete_owner_profile(owner_index)
                continue
            if not owner_name:
                return
            self.record_owner()
            self.remember_current_owner_profile()
        if not rospy.is_shutdown():
            self.face_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
            self.speak(self.all_profiles_recorded_text, wait=True)
            self.publish_status(
                "all_profiles_recorded",
                count=len(self.owner_profiles),
                registered_count=len(self.owner_profiles),
                skipped_count=len(self.skipped_owner_indices),
                skipped_owner_indices=list(self.skipped_owner_indices),
                face_ready=self.face_ready,
            )

    def load_all_owner_profiles(self):
        """把磁盘上已有的主人档案全部读进来（有几份用几份）。

        以前是"缺一份就整体失败"：跳过过第 2 位之后，磁盘上只剩 1、3 两份，
        再想走"不注册、直接用已有档案"就会失败并退回重新注册。
        现在改成：依次尝试 1..owner_count，读得到的就收下，读不到的跳过；
        一份都没有才返回 False（由调用方决定是不是去注册）。
        """
        loaded_profiles = []
        self.owner_profiles = []
        original_owner_name = self.owner_name
        missing_indices = []
        for owner_index in range(1, self.owner_count + 1):
            self.select_owner_profile_path(owner_index)
            self.owner_name = ""
            self.owner_embedding = None
            self.owner_embedding_bank = None
            self.owner_color_embedding = None
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.owner_profile_meta = {}
            if not self.load_owner_profile():
                missing_indices.append(owner_index)
                continue
            self.owner_name = self.owner_profile_meta.get("owner_name", "")
            self.remember_current_owner_profile()
            loaded_profiles.append(self.owner_name or "主人%d" % owner_index)
        if not self.owner_profiles:
            self.owner_name = original_owner_name
            self.publish_status("owner_profiles_missing")
            return False
        if missing_indices:
            rospy.logwarn(
                "第 %s 位没有档案，本轮就用已找到的 %d 位主人",
                "、".join(str(i) for i in missing_indices),
                len(self.owner_profiles),
            )
        self.owner_profiles = sorted(self.owner_profiles, key=lambda item: item["index"])
        self.face_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
        self.publish_status(
            "all_profiles_loaded",
            count=len(self.owner_profiles),
            names=loaded_profiles,
            face_ready=self.face_ready,
        )
        rospy.loginfo("Loaded %d owner profiles: %s", len(self.owner_profiles), ", ".join(loaded_profiles))
        return True

    def save_owner_profile(self, embeddings, sample_meta, color_embeddings=None):
        super().save_owner_profile(embeddings, sample_meta, color_embeddings=color_embeddings)
        face_embeddings = [item["embedding"] for item in self.current_record_face_embeddings]
        if face_embeddings:
            mean_face_embedding = self.normalize_embedding(np.mean(np.vstack(face_embeddings), axis=0))
            if mean_face_embedding is not None:
                face_bank = [mean_face_embedding]
                face_bank.extend(face_embeddings)
                self.owner_face_embedding = mean_face_embedding
                self.owner_face_embedding_bank = np.vstack(face_bank).astype(np.float32)
                with np.load(self.profile_path, allow_pickle=False) as profile_file:
                    npz_payload = {name: profile_file[name] for name in profile_file.files}
                npz_payload["face_embedding"] = self.owner_face_embedding.astype(np.float32)
                npz_payload["face_embedding_bank"] = self.owner_face_embedding_bank
                np.savez(self.profile_path, **npz_payload)
                self.owner_profile_meta["has_face_embedding"] = True
                self.owner_profile_meta["face_samples"] = [
                    {
                        "sample": item["sample"],
                        "pose": item["pose"],
                        "pose_name": item["pose_name"],
                        "image": item["image"],
                        "region": item["region"],
                        "face_count": item["face_count"],
                    }
                    for item in self.current_record_face_embeddings
                ]
        else:
            self.owner_face_embedding = None
            self.owner_face_embedding_bank = None
            self.owner_profile_meta["has_face_embedding"] = False
            self.owner_profile_meta["face_samples"] = []
            if self.face_model_ready:
                rospy.logwarn(
                    "Owner %d profile saved without face embedding; Re-ID fallback remains enabled.",
                    self.current_owner_index,
                )
        self.owner_profile_meta["record_poses"] = [
            {"pose": pose_id, "pose_name": pose_name}
            for pose_id, pose_name, _ in self.owner_record_poses
        ]
        if self.owner_name:
            self.owner_profile_meta["owner_index"] = self.current_owner_index
            self.owner_profile_meta["owner_name"] = self.owner_name
            with open(self.metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(self.owner_profile_meta, metadata_file, ensure_ascii=False, indent=2)
            self.publish_status(
                "profile_named",
                owner_index=self.current_owner_index,
                name=self.owner_name,
                profile_path=self.profile_path,
            )

    def load_owner_profile(self):
        loaded = super().load_owner_profile()
        self.owner_face_embedding = None
        self.owner_face_embedding_bank = None
        if loaded:
            try:
                with np.load(self.profile_path, allow_pickle=False) as profile_file:
                    if "face_embedding" in profile_file.files:
                        self.owner_face_embedding = self.normalize_embedding(profile_file["face_embedding"])
                    if "face_embedding_bank" in profile_file.files:
                        bank = []
                        for vector in np.asarray(profile_file["face_embedding_bank"]):
                            normalized = self.normalize_embedding(vector)
                            if normalized is not None:
                                bank.append(normalized)
                        if bank:
                            self.owner_face_embedding_bank = np.vstack(bank).astype(np.float32)
            except Exception as exc:
                rospy.logwarn("Failed to load owner face embedding from %s: %s", self.profile_path, exc)
            self.face_ready = self.owner_face_embedding_bank is not None
            if not self.owner_name:
                self.owner_name = self.owner_profile_meta.get("owner_name", "")
        return loaded

    def owner_found_message(self):
        name = self.owner_name or self.owner_profile_meta.get("owner_name", "主人")
        return self.format_with_name(self.named_owner_found_text, name)

    def owner_reid_similarity(self, owner_profile, embedding):
        embedding_bank = owner_profile.get("embedding_bank")
        if embedding_bank is not None:
            scores = np.dot(embedding_bank, embedding)
            return float(np.max(scores))
        owner_embedding = owner_profile.get("embedding")
        if owner_embedding is None:
            return -1.0
        return float(np.dot(owner_embedding, embedding))

    def owner_color_similarity(self, owner_profile, crop):
        color_embedding = owner_profile.get("color_embedding")
        if color_embedding is None:
            return None
        query_color_embedding = self.color_hist_embedding(crop)
        if query_color_embedding is None:
            return None
        return float(np.dot(color_embedding, query_color_embedding))

    def owner_face_similarity(self, owner_profile, face_embedding):
        if face_embedding is None:
            return None
        face_bank = owner_profile.get("face_embedding_bank")
        if face_bank is None:
            face_embedding_single = owner_profile.get("face_embedding")
            if face_embedding_single is None:
                return None
            return float(np.dot(face_embedding_single, face_embedding))
        return float(np.max(np.dot(face_bank, face_embedding)))

    def score_owner_match(self, owner_profile, embedding, crop, lying_pose, face_embedding=None):
        reid_score = self.owner_reid_similarity(owner_profile, embedding)
        color_score = self.owner_color_similarity(owner_profile, crop)
        if lying_pose:
            score = self.fused_lie_score(reid_score, color_score)
            face_weight = self.lying_face_identity_weight
        else:
            score = reid_score
            face_weight = self.face_identity_weight
        face_score = self.owner_face_similarity(owner_profile, face_embedding)
        identity_score = score
        if face_score is not None:
            face_score_for_fusion = max(0.0, face_score)
            identity_score = (
                (1.0 - face_weight) * score
                + face_weight * face_score_for_fusion
            )
        return score, reid_score, color_score, face_score, identity_score

    def result_is_match(self, result, default_threshold):
        if result is None:
            return False
        score_margin = result.get("owner_score_margin")
        if (
            len(self.owner_profiles) > 1
            and score_margin is not None
            and float(score_margin) < self.owner_score_margin_threshold
        ):
            return False

        face_score = result.get("face_score")
        reid_score = float(result.get("reid_score", result.get("score", -1.0)))
        lying_pose = bool(result.get("candidate", {}).get("lying_pose", False))
        if lying_pose:
            return PersonReidOwnerTest.result_is_match(result, default_threshold)

        if face_score is not None:
            face_score = float(face_score)
            if face_score >= self.face_accept_threshold and reid_score >= self.face_min_reid_score:
                return True
            if self.face_fast_reject and face_score <= self.face_reject_threshold:
                return False

        return PersonReidOwnerTest.result_is_match(result, default_threshold)

    def evaluate_current_frame(self):
        if not self.owner_profiles:
            return super().evaluate_current_frame()

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
            # 躺姿再往里收一点：把"检测框自带的那圈背景"也去掉，
            # 让喂给 ReID/颜色的图尽量只有人（实测 reid 0.60 -> 0.70）
            shrink = self.lying_crop_shrink if lying_pose else None
            crop, crop_bbox = self.crop_candidate(
                image, candidate, padding=padding, shrink=shrink,
                uniform_padding=lying_pose,
            )
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

        results = []
        face_profiles_ready = any(profile.get("face_embedding_bank") is not None for profile in self.owner_profiles)
        for record in records:
            lying_pose = record["meta"].get("lying_pose", False)
            face_embedding = None
            face_region = ""
            face_count = 0
            face_min_side = 0.0
            face_elapsed_ms = 0.0
            face_reason = "face disabled"
            if self.face_model_ready and face_profiles_ready:
                face_data, face_elapsed_ms, face_reason = self.extract_face_embedding(
                    record["crop"],
                    candidate=record["meta"],
                )
                if face_data is not None:
                    face_embedding = face_data["embedding"]
                    face_region = face_data.get("region", "")
                    face_count = face_data.get("face_count", 0)
                    face_min_side = float(face_data.get("min_side", 0.0) or 0.0)
            for owner_profile in self.owner_profiles:
                variant_scores = []
                for query_index, variant_name in zip(record["query_indexes"], record["variant_names"]):
                    score, reid_score, color_score, face_score, identity_score = self.score_owner_match(
                        owner_profile,
                        embeddings[query_index],
                        record["crop"],
                        lying_pose,
                        face_embedding=face_embedding,
                    )
                    variant_scores.append(
                        (identity_score, score, reid_score, color_score, face_score, variant_name)
                    )
                identity_score, score, reid_score, color_score, face_score, best_variant = max(
                    variant_scores,
                    key=lambda item: item[0],
                )
                if lying_pose:
                    match_threshold = self.lying_match_threshold
                    min_reid_score = self.lying_min_reid_score
                    required_consecutive = self.lying_required_consecutive
                else:
                    match_threshold = self.match_threshold
                    min_reid_score = -1.0
                    required_consecutive = self.match_required_consecutive
                result = {
                    "score": float(score),
                    "identity_score": float(identity_score),
                    "reid_score": float(reid_score),
                    "color_score": None if color_score is None else float(color_score),
                    "face_score": None if face_score is None else float(face_score),
                    "face_region": face_region,
                    "face_count": face_count,
                    "face_min_side": face_min_side,
                    "face_elapsed_ms": face_elapsed_ms,
                    "face_reason": face_reason,
                    "match_threshold": match_threshold,
                    "min_reid_score": min_reid_score,
                    "required_consecutive": required_consecutive,
                    "best_variant": best_variant,
                    "candidate": record["meta"],
                    "crop": record["crop"],
                    "num_candidates": len(records),
                    "owner_index": owner_profile["index"],
                    "owner_name": owner_profile.get("name") or "主人%d" % owner_profile["index"],
                }
                results.append(result)
        if not results:
            return None

        owner_best_scores = {}
        for result in results:
            owner_index = result["owner_index"]
            score = result["identity_score"]
            if owner_index not in owner_best_scores or score > owner_best_scores[owner_index]:
                owner_best_scores[owner_index] = score
        ranked_owner_scores = sorted(owner_best_scores.values(), reverse=True)
        score_margin = None
        if len(ranked_owner_scores) >= 2:
            score_margin = ranked_owner_scores[0] - ranked_owner_scores[1]

        best_result = max(results, key=lambda item: item["identity_score"])
        best_result["owner_score_margin"] = score_margin
        return best_result

    def announce_owner_result(self, result):
        score = result["score"]
        owner_index = result.get("owner_index", self.current_owner_index)
        owner_name = result.get("owner_name", self.owner_name)
        crop_path = self.maybe_save_match_crop(result["crop"], score)
        self.owner_name = owner_name
        message = self.format_with_name(self.named_owner_found_text, owner_name or "主人")
        rospy.loginfo(
            "Owner recognized: index=%s name=%s score=%.3f crop=%s",
            owner_index,
            owner_name or "unknown",
            score,
            crop_path or "not saved",
        )
        self.speak(message, wait=False)
        payload = {
            "event": "owner_recognized",
            "owner_index": owner_index,
            "name": owner_name,
            "score": score,
            "identity_score": result.get("identity_score", score),
            "reid_score": result.get("reid_score", score),
            "color_score": result.get("color_score"),
            "face_score": result.get("face_score"),
            "face_region": result.get("face_region", ""),
            "face_count": result.get("face_count", 0),
            "owner_score_margin": result.get("owner_score_margin"),
            "lying_pose": result.get("candidate", {}).get("lying_pose", False),
            "best_variant": result.get("best_variant", "raw"),
            "crop": crop_path,
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.publish_status(
            "owner_recognized",
            owner_index=owner_index,
            name=owner_name,
            score=score,
            identity_score=result.get("identity_score", score),
            reid_score=result.get("reid_score", score),
            color_score=result.get("color_score"),
            face_score=result.get("face_score"),
            face_region=result.get("face_region", ""),
            face_count=result.get("face_count", 0),
            owner_score_margin=result.get("owner_score_margin"),
            lying_pose=result.get("candidate", {}).get("lying_pose", False),
            best_variant=result.get("best_variant", "raw"),
            crop=crop_path,
        )
        self.last_announce_time = time.time()

    def recognition_loop(self):
        self.publish_status(
            "recognition_started",
            threshold=self.match_threshold,
            owner_count=len(self.owner_profiles) or 1,
            names=[profile.get("name", "") for profile in self.owner_profiles],
            face_model_ready=self.face_model_ready,
            face_ready=self.face_ready,
            face_weight=self.face_identity_weight,
            owner_margin_threshold=self.owner_score_margin_threshold,
        )
        rate = rospy.Rate(max(1.0, 1.0 / self.match_check_interval))
        while not rospy.is_shutdown():
            self.update_yolo_window("识别中")
            result = self.evaluate_current_frame()
            if result is None:
                self.match_consecutive_count = 0
                self.last_match_owner_index = None
                rate.sleep()
                continue

            score = result["score"]
            owner_index = result.get("owner_index", self.current_owner_index)
            owner_name = result.get("owner_name", self.owner_name)
            match_threshold = result.get("match_threshold", self.match_threshold)
            required_consecutive = result.get("required_consecutive", self.match_required_consecutive)
            matched = self.result_is_match(result, self.match_threshold)
            if matched:
                if self.last_match_owner_index == owner_index:
                    self.match_consecutive_count += 1
                else:
                    self.match_consecutive_count = 1
                self.last_match_owner_index = owner_index
            else:
                # 与 scan_for_owner 一致：失败减一而不是清零，
                # 否则躺姿/分数抖动会导致永远凑不满连续次数。
                if self.match_consecutive_count > 0:
                    self.match_consecutive_count -= 1
                if self.match_consecutive_count <= 0:
                    self.last_match_owner_index = None
            self.publish_status(
                "match_score",
                score=score,
                identity_score=result.get("identity_score", score),
                reid_score=result.get("reid_score", score),
                color_score=result.get("color_score"),
                face_score=result.get("face_score"),
                face_region=result.get("face_region", ""),
                face_count=result.get("face_count", 0),
                face_elapsed_ms=result.get("face_elapsed_ms", 0.0),
                owner_score_margin=result.get("owner_score_margin"),
                matched=matched,
                consecutive=self.match_consecutive_count,
                threshold=match_threshold,
                required_consecutive=required_consecutive,
                lying_pose=result.get("candidate", {}).get("lying_pose", False),
                best_variant=result.get("best_variant", "raw"),
                owner_index=owner_index,
                name=owner_name,
            )

            if matched and self.match_consecutive_count >= required_consecutive:
                now = time.time()
                if now - self.last_announce_time >= self.announce_cooldown:
                    crop_path = self.maybe_save_match_crop(result["crop"], score)
                    self.owner_name = owner_name
                    message = self.format_with_name(self.named_owner_found_text, owner_name or "主人")
                    rospy.loginfo(
                        "Owner recognized: index=%s name=%s score=%.3f crop=%s",
                        owner_index,
                        owner_name or "unknown",
                        score,
                        crop_path or "not saved",
                    )
                    self.speak(message, wait=False)
                    payload = {
                        "event": "owner_recognized",
                        "owner_index": owner_index,
                        "name": owner_name,
                        "score": score,
                        "identity_score": result.get("identity_score", score),
                        "reid_score": result.get("reid_score", score),
                        "color_score": result.get("color_score"),
                        "face_score": result.get("face_score"),
                        "face_region": result.get("face_region", ""),
                        "face_count": result.get("face_count", 0),
                        "owner_score_margin": result.get("owner_score_margin"),
                        "lying_pose": result.get("candidate", {}).get("lying_pose", False),
                        "best_variant": result.get("best_variant", "raw"),
                        "crop": crop_path,
                    }
                    self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
                    self.publish_status(
                        "owner_recognized",
                        owner_index=owner_index,
                        name=owner_name,
                        score=score,
                        identity_score=result.get("identity_score", score),
                        reid_score=result.get("reid_score", score),
                        color_score=result.get("color_score"),
                        face_score=result.get("face_score"),
                        face_region=result.get("face_region", ""),
                        face_count=result.get("face_count", 0),
                        owner_score_margin=result.get("owner_score_margin"),
                        lying_pose=result.get("candidate", {}).get("lying_pose", False),
                        best_variant=result.get("best_variant", "raw"),
                        crop=crop_path,
                    )
                    self.last_announce_time = now
                    if not self.action_completed:
                        self.run_owner_action_recognition(result)
                    if self.stop_after_first_match:
                        return
            rate.sleep()

    def run(self):
        self.wait_for_tts()
        if not self.wait_for_asr():
            return
        self.wait_for_camera_inputs()
        self.init_yolo_window()
        self.init_reid_backend()
        self.init_face_recognizer()
        if self.reuse_existing_profile and self.load_all_owner_profiles():
            rospy.loginfo("Loaded existing owner Re-ID profiles from %s", self.profile_dir)
        else:
            self.record_all_owners()
        if not self.owner_profiles:
            if self.owner_embedding is not None:
                self.remember_current_owner_profile()
            else:
                raise RuntimeError("owner profiles are not ready")
        if self.navigate_enabled:
            try:
                self.navigate_to_waypoint(self.waypoint_name)
            except Exception as exc:
                self.stop_base()
                self.speak(self.navigation_failed_text, wait=True)
                self.publish_status("navigation_failed", message=str(exc), waypoint=self.waypoint_name)
                raise
            self.scan_for_owner()
        else:
            self.recognition_loop()


def main():
    rospy.init_node("owner_voice_reid_test")
    node = OwnerVoiceReidTest()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("Owner voice Re-ID test failed: %s", exc)
        node.publish_status(
            "error",
            message=str(exc),
            owner_index=getattr(node, "current_owner_index", 0),
            name=getattr(node, "owner_name", ""),
        )
        raise


if __name__ == "__main__":
    main()
