#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摔倒相关改动的离线回归（不需要机器人、不需要 ROS master）。

覆盖三块：
  A) 方案A + 家具面自证：merge_pose_action_result 的判定（用现场真实读数）
  B) 躯干锚点采样：numpy 关键点不再抛异常（2026-10-11 21:36 那次事故的根因）
  C) 侧向接近 + 人体旋转矩形(OBB)：目标点离身体的安全距离
  F) 2026-10-16 新增：开机硬件自检（/scan + /odom）+ 走近主人的导航看护循环

用法：
  PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:/home/dyp/catkin_ws/devel/lib/python3/dist-packages \
  python3 ~/catkin_ws/src/wpb_task1_owner_search/tools/test_fall_offline.py
"""
import importlib.util
import math
import sys
import threading
import time

import numpy as np
import rospy

rospy.rostime.set_rostime_initialized(True)  # 离线：初始化 rospy 时间，免 ROSInitException

TASK = "/home/dyp/catkin_ws/src/wpb_service_robot/scripts/task_all_owners.py"
ACTION = (
    "/home/dyp/catkin_ws/src/wpb_service_robot/scripts/qwen_action_recognition_node.py"
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FAIL = [0]


def check(name, ok, detail=""):
    print("%s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL[0] += 1


def ground(status, median, p20, local, above=None, confident=None):
    return {
        "status": status,
        "ground_like": status == "ground",
        "ground_confident": (status == "ground") if confident is None else confident,
        "furniture_height_limit": 0.40,
        "ground_height_limit": 0.35,
        "surface_height_median": median,
        "surface_height_p20": p20,
        "local_floor_height_p15": local,
        "height_above_local_floor": above,
    }


def test_plan_a(action_mod):
    """A) 人横躺 + 点云没有（可信）家具证据 → 判摔倒。"""
    cases = [
        ("坏帧(人躺地上, 1.13/1.07/0.64)", ground("elevated", 1.13, 1.07, 0.64, 0.50), "lying", "fallen"),
        ("正常躺地(0.14/0.11/-0.00)", ground("ground", 0.14, 0.11, -0.00, 0.15), "lying", "fallen"),
        ("躺在床上(支撑面0.62/人0.85)", ground("elevated", 0.85, 0.70, 0.62, 0.23), "lying", "lying"),
        # 2026-10-12 现场：躺在床上，但框周围那一圈是**地板**（local=0），
        # 老判据 |median−local|=0.52 > 0.35 被误判成"没有家具证据"→报摔倒。
        ("床上躺(local=地板0.00, 人0.52)", ground("elevated", 0.52, 0.49, 0.00, 0.51), "lying", "lying"),
        ("坐沙发", ground("elevated", 0.60, 0.50, 0.45, 0.15), "sitting", "sitting"),
        ("站在床边", ground("elevated", 0.95, 0.55, 0.00, 0.95), "upright", "unknown"),
        ("点云无信息", ground("unknown", None, None, None), "lying", "fallen"),
    ]
    for name, relation, pose, expected in cases:
        horizontal = pose in ("lying", "sudden_fall", "fallen")
        action, _place, _gf, _el = action_mod.merge_pose_action_result(
            pose, relation, horizontal
        )
        check("A 方案A: " + name, action == expected, "得到 %s 期望 %s" % (action, expected))


def make_task(task_mod):
    task = task_mod.AllOwnersTask.__new__(task_mod.AllOwnersTask)
    task.fall_torso_anchor_enabled = True
    task.fall_torso_keypoint_min_conf = 0.35
    task.fall_body_axis_enabled = True
    task.fall_side_approach_enabled = True
    task.fall_body_axis_min_spread = 0.28
    task.fall_body_axis_max_radius = 1.10
    task.fall_nav_body_clearance = 0.35
    task.fall_nav_side_extend_max = 0.45
    for name in (
        "torso_anchor_from_candidate",
        "body_axis_from_points",
        "body_obb_from_points",
        "fall_standoff_goal_base",
    ):
        setattr(task, name, getattr(task_mod.AllOwnersTask, name).__get__(task))
    return task


def keypoints_numpy():
    """模拟 ultralytics 的真实输出：numpy 数组（当年就是这里炸的）。"""
    points = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros((17,), dtype=np.float32)
    points[5], conf[5] = (430.0, 380.0), 0.9
    points[6], conf[6] = (530.0, 380.0), 0.9
    points[11], conf[11] = (440.0, 430.0), 0.8
    points[12], conf[12] = (520.0, 430.0), 0.8
    return {"bbox": [300, 300, 700, 500], "keypoints": points,
            "keypoint_confidences": conf}


def test_torso_anchor(task):
    """B) 躯干锚点：numpy / NaN / 缺失都不许抛异常。"""
    anchor = task.torso_anchor_from_candidate(keypoints_numpy())
    check(
        "B numpy 关键点能取到躯干中点",
        anchor is not None and abs(anchor[0] - 480.0) < 1e-6 and abs(anchor[1] - 405.0) < 1e-6,
        str(anchor),
    )
    nan_candidate = keypoints_numpy()
    nan_candidate["keypoints"] = np.full((17, 2), np.nan, dtype=np.float32)
    check("B 关键点全 NaN -> None（不抛异常）", task.torso_anchor_from_candidate(nan_candidate) is None)
    low = keypoints_numpy()
    low["keypoint_confidences"] = np.zeros((17,), dtype=np.float32)
    check("B 置信度太低 -> None", task.torso_anchor_from_candidate(low) is None)
    check("B 空候选 -> None", task.torso_anchor_from_candidate(None) is None)


def test_body_geometry(task):
    """C) 身体主方向 / 旋转矩形 / 侧向接近的安全距离。"""
    center = (2.0, 0.0)
    raw = (1.0, 0.25)
    norm = math.hypot(*raw)
    axis = (raw[0] / norm, raw[1] / norm)
    body = [(center[0] + axis[0] * t, center[1] + axis[1] * t)
            for t in [i * 0.04 - 0.8 for i in range(41)]]
    got_axis = task.body_axis_from_points(body, center)
    check(
        "C 身体主方向≈实际方向",
        got_axis is not None and abs(got_axis[0] - axis[0]) < 0.05,
        "得到 %s" % (None if got_axis is None else "(%.2f, %.2f)" % got_axis[:2]),
    )
    obb = task.body_obb_from_points(body, center)
    check(
        "C 旋转矩形：长≈1.44m（身体长 1.6m 去掉 5% 飞点）",
        obb is not None and 1.25 <= 2.0 * obb[4] <= 1.55,
        "长=%.2f 宽=%.2f" % (2.0 * obb[4], 2.0 * obb[5]) if obb else "None",
    )
    gx, gy, face, info = task.fall_standoff_goal_base(center[0], center[1], 0.75, body_xy=body)
    check("C 斜躺时从身体侧面接近", info["mode"] == "side")
    check(
        "C 目标点离身体任何一点 ≥ 0.35m",
        info["min_body_distance"] >= 0.35,
        "实测 %.2f m" % info["min_body_distance"],
    )
    line_goal = (center[0] - 0.75, center[1])
    line_near = min(math.hypot(line_goal[0] - px, line_goal[1] - py) for px, py in body)
    check("C 对照：沿视线老做法会贴到头(%.2f m)" % line_near, line_near < 0.35)
    check(
        "C 朝向=从目标点看向身体中心",
        abs(math.atan2(center[1] - gy, center[0] - gx) - face) < 1e-6,
    )


def test_lying_criteria():
    """D) 躺姿判据：框形状 + 躯干竖直度（新增）+ 躺姿只用 ReID。"""
    enroll = load(
        "/home/dyp/catkin_ws/src/wpb_service_robot/scripts/face_body_enroll_test.py",
        "enroll_offline",
    )
    person = load(
        "/home/dyp/catkin_ws/src/wpb_service_robot/scripts/person_reid_owner_test.py",
        "person_offline",
    )

    class FakeTask(enroll.FaceBodyEnrollTest):
        def __init__(self, verticality_threshold=0.45):
            self.enable_lying_pose_enhancement = True
            self.lying_aspect_ratio_threshold = 1.25
            self.lying_extra_height_ratio = 0.70
            self.lying_extra_aspect_min = 0.55
            self.pose_lying_torso_verticality = verticality_threshold

    task = FakeTask()
    lying_by_shape = task.is_lying_candidate({"bbox": [0, 0, 300, 100], "aspect_ratio": 3.0, "height_ratio": 0.20})
    check("D 横条框 -> 判躺姿（形状）", lying_by_shape is True)
    # 现场实际出现的那一帧：宽高比 0.30、高度占比 0.45（形状看不出），但姿态说是 lying
    upright_like = task.is_lying_candidate({"bbox": [0, 0, 100, 300], "aspect_ratio": 0.30, "height_ratio": 0.45})
    check("D 形状看不出、也没有竖直度 -> 不判躺姿", upright_like is False)
    by_verticality = task.is_lying_candidate(
        {"bbox": [0, 0, 100, 300], "aspect_ratio": 0.30, "height_ratio": 0.45, "torso_verticality": 0.30}
    )
    check("D 躯干竖直度 0.30 < 0.45 -> 判躺姿（新增判据）", by_verticality is True)
    standing = task.is_lying_candidate(
        {"bbox": [0, 0, 100, 300], "aspect_ratio": 0.40, "height_ratio": 0.90, "torso_verticality": 0.95}
    )
    check("D 站着的人（竖直度 0.95）不判躺姿", standing is False)

    # 躺姿融合：颜色权重必须为 0（只用 ReID）
    base = person.PersonReidOwnerTest.__new__(person.PersonReidOwnerTest)
    base.lying_reid_weight, base.lying_color_weight = 1.0, 0.0
    fused = base.fused_lie_score(0.595, 0.136)
    check("D 躺姿融合分 = ReID（颜色不参与）", abs(fused - 0.595) < 1e-9, "得到 %.3f" % fused)


def test_action_speech():
    """E) 动作播报文案：主人{姓名}正在{房间}{动作}。"""
    fb = load(
        "/home/dyp/catkin_ws/src/wpb_service_robot/scripts/feedback_after_recognition.py",
        "feedback_offline",
    )
    got = fb.feedback_for_action("sitting", "unknown", "张三", room="客厅").speech
    check("E 坐着 + 客厅", got == "主人张三正在客厅坐着", got)
    got = fb.feedback_for_action("lying", "unknown", "张三", room="卧室").speech
    check("E 躺着 + 卧室", got == "主人张三正在卧室躺着", got)
    got = fb.feedback_for_action("fallen", "floor", "张三", room="客厅").speech
    check("E 摔倒用「在客厅摔倒了」", got == "主人张三在客厅摔倒了", got)
    got = fb.feedback_for_action("waving", "unknown", "张三").speech
    check("E 没有房间名时退回老格式", got == "主人张三正在挥手", got)


class _StubMoveBase:
    def __init__(self, state_fn):
        self.state_fn = state_fn
        self.cancelled = False

    def wait_for_server(self, _duration):
        return True

    def send_goal(self, _goal):
        pass

    def get_state(self):
        return self.state_fn()

    def cancel_goal(self):
        self.cancelled = True


class _WatchdogStub:
    """只实现 navigate_fall_goal_with_guard 用到的接口。"""

    def __init__(self, state_fn, travelled_fn, speed_fn, guard_fn, owner_match=True):
        from actionlib_msgs.msg import GoalStatus  # noqa: F401

        self.move_base = _StubMoveBase(state_fn)
        self.move_base_server_timeout = 3.0
        self.fall_nav_watchdog_enabled = True
        self.fall_nav_frame = "map"
        self.fall_nav_guard_distance = 0.40
        self.fall_nav_guard_is_arrival = True
        self.fall_nav_slow_finish_enabled = True
        self.fall_nav_slow_finish_speed = 0.03
        self.fall_nav_slow_finish_cycles = 3
        self.fall_nav_slow_finish_tolerance = 0.20
        self.fall_nav_arrival_tolerance = 0.50
        self.fall_nav_stuck_timeout = 2.0
        self.fall_nav_stuck_min_progress = 0.05
        self.fall_nav_stuck_speed = 0.025
        self.fall_nav_stuck_angular_speed = 0.05
        self.fall_nav_odom_speed_max_age = 0.6
        self.start_time = time.time()
        self._travelled_fn = travelled_fn
        self._speed_fn = speed_fn
        self._guard_fn = guard_fn
        self._owner_match = owner_match

    def stop_base(self):
        pass

    def clear_move_base_costmaps(self, _reason):
        return True

    def update_yolo_window(self, _label):
        pass

    def _fall_console(self, _message):
        pass

    def robot_pose_in_map(self, timeout=1.0):
        return 0.0, 0.0, 0.0, 0.0

    def get_latest_odom_xy(self):
        return self._travelled_fn(), 0.0, 0.0

    def get_latest_odom_speeds(self, max_age=None):
        linear, angular = self._speed_fn()
        return linear, angular, 0.0

    def scan_forward_min_distance(self):
        return self._guard_fn()

    def guard_obstacle_is_owner(self, _bbox):
        return self._owner_match


def test_hardware_gate(task_mod):
    """F1) 开机硬件自检：/scan 和 /odom 都没数据时必须拦下来（不盲开）。"""

    class HwStub:
        def __init__(self, scan_stamp, odom_stamp, require=True, timeout=1.0):
            self.lock = threading.Lock()
            self.scan_topic = "/scan"
            self.odom_topic = "/odom"
            self.require_scan = require
            self.require_odom = require
            self.hardware_check_timeout = timeout
            self._scan_msg_stamp = scan_stamp
            self.latest_odom_stamp = odom_stamp
            self.stopped = 0

        def ensure_scan_subscription(self):
            return True

        def stop_base(self):
            self.stopped += 1

        def is_topic_advertised(self, topic_name):
            return task_mod.AllOwnersTask.is_topic_advertised(self, topic_name)

        def describe_missing_topic(self, topic_name, label):
            return task_mod.AllOwnersTask.describe_missing_topic(
                self, topic_name, label
            )

    now = time.time()
    check(
        "F1 硬件自检：scan/odom 都有数据 → 通过",
        task_mod.AllOwnersTask.check_hardware(HwStub(now, now)) is True,
    )
    missing = HwStub(0.0, 0.0, timeout=1.0)
    check(
        "F1 硬件自检：都没有数据 → 拒绝启动并停车",
        task_mod.AllOwnersTask.check_hardware(missing) is False
        and missing.stopped == 1,
    )
    check(
        "F1 硬件自检：_require_scan/odom=false → 直接放行",
        task_mod.AllOwnersTask.check_hardware(
            HwStub(0.0, 0.0, require=False)
        )
        is True,
    )


def test_approach_watchdog(task_mod):
    """F2) 走近主人的导航看护循环：慢停 / 卡死 / 假成功 / 护栏。"""
    from actionlib_msgs.msg import GoalStatus
    from move_base_msgs.msg import MoveBaseGoal

    def make_goal():
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = 1.0
        goal.target_pose.pose.position.y = 0.0
        goal.target_pose.pose.orientation.w = 1.0
        return goal

    def run(stub, timeout=3.0):
        return task_mod.AllOwnersTask.navigate_fall_goal_with_guard(
            stub, make_goal(), timeout, "测试"
        )

    s_normal = _WatchdogStub(
        lambda: GoalStatus.SUCCEEDED
        if time.time() - s_normal.start_time > 0.6
        else GoalStatus.ACTIVE,
        lambda: min(1.0, (time.time() - s_normal.start_time) * 2.0),
        lambda: (0.5, 0.0),
        lambda: None,
    )
    ok, reason = run(s_normal)
    check("F2 正常走到位 → 成功", ok, reason)

    s_fake = _WatchdogStub(
        lambda: GoalStatus.SUCCEEDED, lambda: 0.0, lambda: (0.0, 0.0), lambda: None
    )
    ok, reason = run(s_fake)
    check("F2 move_base 假成功（没动）→ 判失败", (not ok) and ("还差" in reason), reason)

    s_stuck = _WatchdogStub(
        lambda: GoalStatus.ACTIVE, lambda: 0.0, lambda: (0.0, 0.0), lambda: None
    )
    ok, reason = run(s_stuck, timeout=6.0)
    check("F2 卡死不动 → 提前取消", (not ok) and ("卡住" in reason), reason)

    s_slow = _WatchdogStub(
        lambda: GoalStatus.ACTIVE,
        lambda: 0.85 if time.time() - s_slow.start_time > 0.2 else 0.0,
        lambda: (0.01, 0.0),
        lambda: None,
    )
    ok, reason = run(s_slow)
    check("F2 剩余很近且已慢停 → 当作到位", ok and ("慢停" in reason), reason)

    s_guard_true = _WatchdogStub(
        lambda: GoalStatus.ACTIVE,
        lambda: 0.2,
        lambda: (0.3, 0.0),
        lambda: 0.35,
        owner_match=True,
    )
    ok, reason = run(s_guard_true)
    check("F2 护栏触发且确认是主人 → 当作到位", ok, reason)

    s_guard_false = _WatchdogStub(
        lambda: GoalStatus.ACTIVE,
        lambda: 0.2,
        lambda: (0.3, 0.0),
        lambda: 0.35,
        owner_match=False,
    )
    ok, reason = run(s_guard_false)
    check("F2 护栏触发但不是主人 → 判真障碍", (not ok) and ("障碍" in reason), reason)

    s_turn = _WatchdogStub(
        lambda: GoalStatus.ACTIVE, lambda: 0.0, lambda: (0.0, 0.6), lambda: None
    )
    ok, reason = run(s_turn)
    check(
        "F2 原地转向不算卡死（角速度仍在动）",
        (not ok) and ("卡住" not in reason),
        reason,
    )


def test_rotation_guard(task_mod):
    """H) 原地旋转护栏：全向最近距离贴到车体就停转（2026-10-16）。

    背景：转圈找人 / 转正都是裸发 /cmd_vel、只看里程计，车体旋转会横扫**侧面**
    的椅子腿，而"正前方 ±25°"那条护栏管不到。新护栏看全向最近距离：
      H1 scan_callback 同时给出正前方与全向两个最近距离；
      H2 转正（rotate_toward_bearing）遇护栏 → 不转、返回 False；
      H3 护栏没数据（None）时转正照常转（不能因为加护栏把正常功能掐死）；
      H4 转圈找人（scan_for_owner_for_target）遇护栏 → 不发旋转指令。
    """
    import threading
    import types

    class ScanMsg:
        pass

    # ---- H1：雷达双通道 ----
    stub = task_mod.AllOwnersTask.__new__(task_mod.AllOwnersTask)
    stub.lock = threading.Lock()
    stub.scan_forward_half_angle_deg = 25.0
    stub._scan_forward_min = None
    stub._scan_any_min = None
    stub._scan_stamp = 0.0
    stub._scan_msg_stamp = 0.0
    msg = ScanMsg()
    msg.range_min = 0.25
    msg.range_max = 12.0
    msg.angle_min = -math.pi
    msg.angle_increment = math.pi / 180.0
    msg.ranges = [13.0] * 360  # >3.0 视为无效（贴图上限）
    msg.ranges[180] = 1.50     # 正前方 1.50 m（±25° 内）
    msg.ranges[240] = 0.28     # 左前 60°（窗口外）：只有全向看得见
    task_mod.AllOwnersTask.scan_callback(stub, msg)
    forward = task_mod.AllOwnersTask.scan_forward_min_distance(stub)
    any_min = task_mod.AllOwnersTask.scan_any_min_distance(stub)
    check(
        "H1 雷达双通道：正前方 1.50 m / 全向 0.28 m",
        abs((forward or 0.0) - 1.50) < 1e-6 and abs((any_min or 0.0) - 0.28) < 1e-6,
        "forward=%s any=%s" % (forward, any_min),
    )

    def make_turn_stub(guard_value, timeout=0.4):
        s = task_mod.AllOwnersTask.__new__(task_mod.AllOwnersTask)
        s.fall_snapshot_align_timeout = timeout
        s.fall_snapshot_bearing_tolerance = 0.10
        s.fall_snapshot_turn_speed = 0.35
        s.scan_turn_guard_distance = 0.30
        s.get_latest_yaw = lambda: 0.0
        s.scan_any_min_distance = lambda: guard_value
        s.published = []
        s.cmd_pub = types.SimpleNamespace(
            publish=lambda twist: s.published.append(twist)
        )
        s.stop_base = lambda: s.published.append("stop")
        return s

    # ---- H2：贴到车体 → 不转 ----
    stub = make_turn_stub(0.28)
    ok = task_mod.AllOwnersTask.rotate_toward_bearing(stub, 1.0, label="测试")
    twists = [item for item in stub.published if not isinstance(item, str)]
    check(
        "H2 转正遇护栏（全向 0.28 m ≤ 0.30 m）→ 停转、返回 False",
        (ok is False) and twists == [] and ("stop" in stub.published),
        "ok=%s twists=%d" % (ok, len(twists)),
    )

    # ---- H3：没有护栏数据时照常转（防止"加了护栏把功能掐死"）----
    stub = make_turn_stub(None)
    task_mod.AllOwnersTask.rotate_toward_bearing(stub, 1.0, label="测试")
    twists = [item for item in stub.published if not isinstance(item, str)]
    check(
        "H3 护栏无数据（None）→ 转正照常发 /cmd_vel",
        len(twists) > 0,
        "twists=%d" % len(twists),
    )

    # ---- H4：转圈找人遇护栏 → 不发旋转指令 ----
    def make_scan_stub(guard_value):
        s = task_mod.AllOwnersTask.__new__(task_mod.AllOwnersTask)
        s.scan_after_arrival_delay = 0.0
        s.scan_timeout = 0.4
        s.scan_total_angle = math.pi
        s.scan_step_angle = 0.18
        s.scan_angular_speed = -0.25
        s.scan_pause = 0.05
        s.scan_max_yaw_step = 0.30
        s.match_check_interval = 0.10
        s.match_threshold = 0.5
        s.match_required_consecutive = 2
        s.scan_turn_guard_distance = 0.30
        s.get_latest_yaw = lambda: 0.0
        s.scan_any_min_distance = lambda: guard_value
        # 相机看门狗在离线里直接放行（真实实现要相机订阅等一堆依赖）
        s.hold_until_camera_recovers = lambda label: True
        s.evaluate_current_frame = lambda: None
        s.result_is_match = lambda *a, **k: False
        s.update_yolo_window = lambda *a, **k: None
        s.publish_status = lambda *a, **k: None
        s.speak = lambda *a, **k: None
        s.published = []
        s.cmd_pub = types.SimpleNamespace(
            publish=lambda twist: s.published.append(twist)
        )
        s.stop_base = lambda: s.published.append("stop")
        return s

    stub = make_scan_stub(0.25)
    task_mod.AllOwnersTask.scan_for_owner_for_target(stub, not_found_text="")
    twists = [item for item in stub.published if not isinstance(item, str)]
    check(
        "H4 转圈找人遇护栏（全向 0.25 m）→ 一次旋转指令都不发",
        twists == [],
        "twists=%d" % len(twists),
    )

    stub = make_scan_stub(None)
    task_mod.AllOwnersTask.scan_for_owner_for_target(stub, not_found_text="")
    twists = [item for item in stub.published if not isinstance(item, str)]
    check(
        "H4 对照：护栏没数据时转圈照常发指令",
        len(twists) > 0,
        "twists=%d" % len(twists),
    )


def test_score_and_face_gates(task_mod, action_mod):
    """I) 2026-10-17 两处识别改动：
      I1 分数分叉用"最近 N 帧中位数"，单帧噪声不再触发"分不清是谁"；
      I2 N=1 时退回单帧（可关闭）；
      I3 框里多张脸时选"最大/最居中"的那张（面积为主、居中为副）。
    """
    import types

    # ---- I1/I2：时序聚合 ----
    stub = task_mod.AllOwnersTask.__new__(task_mod.AllOwnersTask)
    stub.owner_score_margin_frames = 3
    stub._owner_score_history = {}
    margin1, source1, med1 = task_mod.AllOwnersTask.smoothed_owner_score_margin(
        stub, {1: 0.90, 2: 0.60}, names={1: "甲", 2: "乙"}
    )
    # 第 2 帧：甲偶然掉到 0.62（单帧 margin 只有 0.02，会误判"分不清"）
    margin2, source2, med2 = task_mod.AllOwnersTask.smoothed_owner_score_margin(
        stub, {1: 0.62, 2: 0.60}, names={1: "甲", 2: "乙"}
    )
    check(
        "I1 时序聚合：单帧噪声 0.02 不再触发（中位 margin 仍 0.16）",
        abs(margin2 - 0.16) < 1e-6 and source2.startswith("median"),
        "margin=%.3f source=%s medians=%s" % (margin2, source2, med2),
    )
    check(
        "I1 首帧：只有 1 帧样本时按单帧 margin 算",
        abs(margin1 - 0.30) < 1e-6 and source1 == "frame",
        "margin=%.3f source=%s" % (margin1, source1),
    )
    # 真实的"两个人都接近"：连续 3 帧都接近 → 中位数仍然接近（该拦还是要拦）
    task_mod.AllOwnersTask.smoothed_owner_score_margin(stub, {1: 0.61, 2: 0.60})
    margin3, _source3, _ = task_mod.AllOwnersTask.smoothed_owner_score_margin(
        stub, {1: 0.615, 2: 0.60}
    )
    check(
        "I1 真实接近（连续多帧）→ margin 仍然很小（闸门照常生效）",
        margin3 < 0.04,
        "margin=%.3f" % margin3,
    )

    stub_off = task_mod.AllOwnersTask.__new__(task_mod.AllOwnersTask)
    stub_off.owner_score_margin_frames = 1
    stub_off._owner_score_history = {1: [0.9], 2: [0.5]}
    margin_off, source_off, _ = task_mod.AllOwnersTask.smoothed_owner_score_margin(
        stub_off, {1: 0.62, 2: 0.60}
    )
    check(
        "I2 N=1 → 退回单帧，且清空历史",
        margin_off is None and source_off == "frame" and stub_off._owner_score_history == {},
        "margin=%s source=%s" % (margin_off, source_off),
    )

    # ---- I3：多脸时选"最大/最居中" ----
    def face(bbox, det=0.9):
        return types.SimpleNamespace(bbox=list(bbox), det_score=det)

    big = face([10, 10, 60, 60])            # 面积 2500（偏左上）
    small_center = face([90, 90, 110, 110])  # 面积 400（正中）
    picked = task_mod.AllOwnersTask.select_largest_face(
        [small_center, big], image_shape=(200, 200)
    )
    check(
        "I3 多脸：面积大的优先（仍是主人框里那张大脸）",
        picked is big,
        "picked=%s" % (getattr(picked, "bbox", None),),
    )
    big_off = face([0, 0, 50, 50])          # 面积 2500，贴边
    big_center = face([80, 80, 130, 130])   # 同面积，居中
    picked2 = task_mod.AllOwnersTask.select_largest_face(
        [big_off, big_center], image_shape=(200, 200)
    )
    check(
        "I3 多脸：面积相近时优先居中那张",
        picked2 is big_center,
        "picked=%s" % (getattr(picked2, "bbox", None),),
    )

    # ---- I4：3D 腿部几何（诊断用）----
    # 光学系 x右 y下 z前；height_fn 模拟"相机离地 1.0 m"
    height_fn = lambda point: 1.0 - float(point[1])
    seated = action_mod.PoseActionRecognitionNode.leg_geometry_from_3d(
        (0.0, 0.55, 1.00),   # 髋：离地 0.45，前方 1.0
        (0.30, 0.58, 1.10),  # 膝：离地 0.42，横向 0.3、前方 1.1（大腿近水平）
        (0.32, 0.92, 1.05),  # 踝：离地 0.08
        height_fn=height_fn,
    )
    check(
        "I4 坐姿 3D：大腿落差小、倾角小（≈5°）",
        abs(seated.get("thigh_drop", 9) - 0.03) < 1e-6
        and seated.get("thigh_angle_deg", 90) < 10.0
        and abs(seated.get("hip_height", 0) - 0.45) < 1e-6
        and abs(seated.get("ankle_height", 0) - 0.08) < 1e-6,
        "seated=%s" % (seated,),
    )
    standing = action_mod.PoseActionRecognitionNode.leg_geometry_from_3d(
        (0.0, 0.10, 1.00),   # 髋：离地 0.90
        (0.02, 0.55, 1.02),  # 膝：离地 0.45（大腿近竖直）
        height_fn=height_fn,
    )
    check(
        "I4 站姿 3D：大腿落差大、倾角≈90°",
        standing.get("thigh_drop", 0) > 0.4
        and standing.get("thigh_angle_deg", 0) > 80.0,
        "standing=%s" % (standing,),
    )
    missing = action_mod.PoseActionRecognitionNode.leg_geometry_from_3d(
        None, (0.0, 0.5, 1.0), height_fn=height_fn
    )
    check("I4 缺髋/膝 → 返回空（跳过该帧）", missing == {}, "missing=%s" % (missing,))

    # ---- I5：3D 坐姿兜底阈值（只在 2D unknown 时启用）----
    accepts = action_mod.PoseActionRecognitionNode.sitting_3d_accepts
    check(
        "I5 3D 坐姿：倾角 12°/落差 0.05/水平 0.31 → 成立",
        accepts(12.0, 0.05, 0.31, 0.95, 0.53) is True,
        "seated",
    )
    check(
        "I5 站姿不会误判：倾角 86°/落差 0.45/水平 0.03 → 不成立",
        accepts(86.0, 0.45, 0.03, 1.00, 0.70) is False,
        "standing",
    )
    check(
        "I5 躺姿（躯干 0.30）→ 不成立",
        accepts(12.0, 0.05, 0.31, 0.30, 0.53) is False,
        "lying torso",
    )
    check(
        "I5 正对坐姿（膝髋比 0.68，2D 比值偏大）→ 仍成立",
        accepts(23.0, 0.14, 0.32, 1.00, 0.68) is True,
        "frontal seated",
    )
    check(
        "I5 坐床/沙发（倾角 55°、落差 0.26、水平 0.17）→ 成立",
        accepts(55.0, 0.26, 0.17, 1.00, 0.74) is True,
        "high-seat seated",
    )
    check(
        "I5 站姿（66°/0.36/0.16）→ 不成立（落差挡住）",
        accepts(66.0, 0.36, 0.16, 1.00, 0.65) is False,
        "standing drop guard",
    )
    check(
        "I5 站姿（75°/0.36/0.10）→ 不成立（水平挡住）",
        accepts(75.0, 0.36, 0.10, 1.00, 0.65) is False,
        "standing horizontal guard",
    )
    check(
        "I5 缺躯干 → 不成立；缺膝髋比（比值不再当门槛）→ 仍成立",
        accepts(12.0, 0.05, 0.31, None, 0.53) is False
        and accepts(12.0, 0.05, 0.31, 0.95, None) is True,
        "missing inputs",
    )


def test_approach_owner_navigate(task_mod):
    """G) 挥手靠近改成"导航式"：先 move_base 绕障，再点云闭环收尾（2026-10-16）。

    背景：老 approach_owner 是纯相机闭环 + 直发 /cmd_vel，不查 /scan 也不走代价
    地图，主人站在椅子后面挥手时机器人一路撞椅子腿。新实现必须满足：
      G1 距离远 → 把"主人前方 standoff(1.0 m)"的点交给 move_base；
      G2 收尾段护栏触发（前方有障碍）→ 返回失败，不再前进、也不再里程计补齐；
      G3 move_base 走不通 → 返回失败（调用方原地不动直接问答，不会硬开）；
      G4 已经站在 standoff 附近 → 不导航，直接收尾。
    """
    snapshot_template = {
        "forward": 2.00,
        "lateral": 0.10,
        "distance": 2.0012,
        "bearing": 0.05,
        "samples": 3,
        "points": 42,
        "points_xy": [],
        "body_xy": [],
        "source": "band",
        "bbox": [10, 10, 40, 90],
    }

    def make_stub(distance=2.0012, nav_ok=True, guard=False, drive_results=None,
                  creep_max=0.0, finish_retries=1):
        stub = task_mod.AllOwnersTask.__new__(task_mod.AllOwnersTask)
        stub.approach_enabled = True
        stub.approach_navigate_enabled = True
        stub.approach_standoff_distance = 1.0
        stub.approach_min_distance = 0.6
        stub.approach_nav_stage_standoff_distance = 0.0
        stub.approach_nav_timeout = 30.0
        stub.approach_nav_obstacle_retries = 1
        stub.approach_nav_obstacle_shift = 0.35
        stub.approach_navigate_fallback_camera = False
        stub.approach_face_first_enabled = False
        stub.approach_face_first_max_angle = 0.35
        stub.approach_creep_max_distance = creep_max
        stub.approach_finish_retries = finish_retries
        stub.approach_owner_band_low = 0.15
        stub.approach_owner_band_high = 0.85
        stub.approach_camera_forward_offset = 0.05
        stub.fall_nav_distance_tolerance = 0.20
        stub.fall_nav_frame = "map"
        stub.fall_nav_plan_block = False
        stub.fall_body_axis_enabled = False
        stub.fall_side_approach_enabled = False
        stub.fall_scan_guard_distance = 0.40
        stub.sent = []
        stub.drove = []
        stub.creeped = []
        stub.ensure_approach_pointcloud = lambda: True
        stub.ensure_scan_subscription = lambda: True
        stub.fall_speak_approach_start_once = lambda: True
        stub.robot_pose_in_map = lambda timeout=1.0: (1.0, 2.0, 0.0, 0.1)
        stub.fall_nav_plan_ok = lambda start, goal: (True, "预检通过")
        stub.publish_fall_debug_markers = lambda *a, **k: None
        stub._approach_console = lambda *a, **k: None

        def fake_snapshot(*args, **kwargs):
            data = dict(snapshot_template)
            data["distance"] = distance
            return data

        stub.capture_fallen_owner_snapshot = fake_snapshot

        def fake_send(goal, timeout, label, owner_bbox=None):
            stub.sent.append(
                (goal.target_pose.pose.position.x, goal.target_pose.pose.position.y)
            )
            return nav_ok, ("" if nav_ok else "前方 0.30 m 有障碍（护栏 0.40 m），取消导航")

        stub.navigate_fall_goal_with_guard = fake_send

        queue = list(drive_results) if drive_results else None

        def fake_drive(bbox, label="主人", stop_distance=None):
            stub.drove.append(stop_distance)
            if queue:
                value = queue.pop(0) if len(queue) > 1 else queue[0]
                return (value[0], value[1])
            return (1.05, "scan_guard" if guard else "cloud_stop")

        stub.drive_forward_until_cloud_close = fake_drive

        def fake_creep(distance, label="主人", scan_stop_distance=None):
            stub.creeped.append(distance)
            return True, distance

        stub.creep_forward_by_odometry = fake_creep
        return stub

    # G1：主人 2.0 m 在正前方 → 中间点应落在"主人前方 1.0 m"（机器人 map=(1,2) yaw=0）
    stub = make_stub()
    ok = task_mod.AllOwnersTask.approach_owner_navigate(stub, "李华", [10, 10, 40, 90])
    goal = stub.sent[0] if stub.sent else None
    check(
        "G1 挥手靠近：先交 move_base，目标在主人前方 1.0 m",
        bool(ok)
        and goal is not None
        and 1.90 < goal[0] < 2.20
        and 1.95 < goal[1] < 2.15,
        "goal=%s" % (goal,),
    )
    check(
        "G1 挥手靠近：收尾段交给带护栏的点云闭环",
        stub.drove == [1.0],
        "stop_distance=%s" % (stub.drove,),
    )

    # G2：收尾段护栏触发（前面真有障碍）→ 判失败，不前进、不补齐
    stub = make_stub(guard=True)
    ok = task_mod.AllOwnersTask.approach_owner_navigate(stub, "李华", [10, 10, 40, 90])
    check(
        "G2 收尾护栏触发 → 停住判失败，不做里程计补齐",
        (not ok) and stub.creeped == [],
        "ok=%s creeped=%s" % (ok, stub.creeped),
    )

    # G3：move_base 走不通 → 判失败，且**不**进入收尾段（调用方会原地问答）
    stub = make_stub(nav_ok=False)
    ok = task_mod.AllOwnersTask.approach_owner_navigate(stub, "李华", [10, 10, 40, 90])
    check(
        "G3 move_base 走不通 → 判失败且不硬开",
        (not ok) and stub.drove == [],
        "ok=%s drove=%s" % (ok, stub.drove),
    )

    # G4：已经在 standoff 附近 → 跳过导航，直接收尾
    stub = make_stub(distance=1.05)
    ok = task_mod.AllOwnersTask.approach_owner_navigate(stub, "李华", [10, 10, 40, 90])
    check(
        "G4 已在 standoff 附近 → 不导航，直接收尾",
        bool(ok) and stub.sent == [] and stub.drove == [1.0],
        "ok=%s sent=%s" % (ok, stub.sent),
    )

    # G5：收尾第一次闭环没到位（还差 0.35 m）→ 再用**点云闭环**补一次，不盲走
    stub = make_stub(
        drive_results=[(1.35, "timeout"), (1.02, "cloud_stop")],
    )
    ok = task_mod.AllOwnersTask.approach_owner_navigate(stub, "李华", [10, 10, 40, 90])
    check(
        "G5 收尾差 0.35 m → 用点云闭环再补一次（不做里程计盲走）",
        bool(ok) and stub.drove == [1.0, 1.0] and stub.creeped == [],
        "ok=%s drove=%s creeped=%s" % (ok, stub.drove, stub.creeped),
    )

    # G6：闭环一直没能确认到距离 → 默认**不盲走**，返回 False（原地进入问答）
    stub = make_stub(
        drive_results=[(1.60, "timeout")],
    )
    ok = task_mod.AllOwnersTask.approach_owner_navigate(stub, "李华", [10, 10, 40, 90])
    check(
        "G6 闭环没到位 → 默认不盲走，判失败",
        (not ok) and stub.creeped == [] and len(stub.drove) == 2,
        "ok=%s drove=%s creeped=%s" % (ok, stub.drove, stub.creeped),
    )

    # G7：显式打开盲走（_approach_creep_max_distance>0）时才做里程计补齐
    stub = make_stub(
        drive_results=[(1.60, "timeout")],
        creep_max=0.30,
        finish_retries=0,
    )
    task_mod.AllOwnersTask.approach_owner_navigate(stub, "李华", [10, 10, 40, 90])
    check(
        "G7 显式打开盲走时才补齐（且不超过开关值）",
        stub.creeped and abs(stub.creeped[0] - 0.30) < 1e-6,
        "creeped=%s" % (stub.creeped,),
    )


def main():
    task_mod = load(TASK, "task_all_owners_offline")
    action_mod = load(ACTION, "qwen_action_offline")
    test_plan_a(action_mod)
    task = make_task(task_mod)
    test_torso_anchor(task)
    test_body_geometry(task)
    test_lying_criteria()
    test_action_speech()
    test_hardware_gate(task_mod)
    test_approach_watchdog(task_mod)
    test_approach_owner_navigate(task_mod)
    test_rotation_guard(task_mod)
    test_score_and_face_gates(task_mod, action_mod)
    print("")
    print("失败 %d 项" % FAIL[0])
    return 1 if FAIL[0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
