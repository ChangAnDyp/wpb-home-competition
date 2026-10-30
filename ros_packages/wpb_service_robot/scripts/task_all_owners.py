#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""总任务：多位主人注册 → 逐位寻找 → 认人报姓名 → 报动作。

流程（对应比赛"识别主人 + 判定行为"那条主线）
------------------------------------------------
  第 0 步  等语音 / 相机 / 模型就绪
  第 1 步  主人注册（默认 3 位：姓名 → 确认 → 人脸 → 身形）
           第 2 位起可以语音说"跳过"；已注册过时可用 ~reuse_existing_profile 跳过
  第 2 步  对每一位主人循环：
             2.1  导航到第 1 个房间（~search_waypoints[0]）
             2.2  原地转一圈，认"还没找到的那几位"里的任意一位
             2.3  把主人居中到画面中间
             2.4  播报"找到主人X了，现在开始检测您的动作"
             2.5  调动作识别节点，播报"主人X正在Y"
             2.6  播报之后的实际反应：
                   挥手   → 走近 → 问"需要我帮您做什么" → 复述主人的原话
                   坐/躺  → 问要不要操作电器开关 → 播报开关的开/关状态
                   摔倒   → 走近（0.8 米）→ 伸机械臂 → 保持 → 收回
             2.7  处理完接着把这一圈转完 —— 同一间屋里站着两位主人也能一次扫出来
                  转完这一圈还有没找到的，才换下一个房间
             2.8  离开这个房间前，把顺手累积到的垃圾播报出来
                   （"我找到了垃圾：空瓶子。" —— 只找 + 播报，不抓不入桶）
  第 3 步  汇总播报本次检查结果，然后导航回出口（~exit_waypoint_name）

设计说明
--------
* 本文件是编排层：注册 / 认人 / 扫描全部复用父类 FaceBodyEnrollTest 的方法，
  动作识别交给独立节点 qwen_action_recognition_node.py（两个进程之间走话题），
  所以这里不重复实现任何算法。
* "主人X正在Y"这句话由 feedback_after_recognition.py 的纯逻辑拼出来。
  为了避免播两遍，动作识别节点自己的那句要关掉（~action_speak_enabled，
  本文件默认关；老流程 ③d 默认仍是开，行为不变）。
* "播报之后的实际反应"（规则里"根据行为/指令做出正确人机交互"那 150 分）
  由 dispatch_next_step() 按 next_step 分派，目前接好了两条：
    走近主人（approach_owner）→ 问"需要我帮您做什么" → 把 ASR 原话复述回去
    电器开关（ask_switch_instruction / operate_switch）→ **只播报状态、不动机械臂**
  摔倒协助（approach_and_assist）→ 走近到 0.8 米 → 伸机械臂 → 保持 4 秒 → 收回。
  现在 dispatch 里**四条分支都接上了**，文件末尾只剩 check_hardware 一个占位。
* 地面垃圾（规则 2.6-6，"找到垃圾"3×20 分）：只做"找到 + 播报"，不做抓取和入桶。
  判定与累积逻辑是从 cpy 的工作区（catkin_ws1）搬过来的，数值保持原样。
  检测来源是独立的 YOLO-World 节点（开放词表，见 launch 配置 ⑭），
  话题用 ~trash_detections_topic 单独接 —— 不动主人识别那条链路。
* 导航：真实调用 move_base 走到航点（航点文件 ~waypoint_file，默认
  ~/waypoints.xml）。地图和航点都已经在包里（maps/、config/），起栈时用
  ⑦/⑦b 的 map:=... 指到新包那份地图即可。
  · 导航连续失败到 ~navigation_failure_limit 次就自动放弃导航、退回"就地找"，
    避免定位坏掉时把 8 分钟全烧在路上（每个房间干等一个导航超时）。
  · 时间预算：~competition_total_seconds 内留出 ~competition_exit_reserve_seconds
    给"回出口"，到点就不再接新房间，直接收尾。
  · 不想要导航（没建图 / 调试算法）就传 _navigate_enabled:=false，回到原地转圈。

常用参数
--------
  ~owner_count               注册几位主人（默认 3）
  ~reuse_existing_profile    跳过注册，直接用已有档案（调试用）
  ~navigate_enabled          是否启用导航（默认 false = 原地转圈）
  ~scan_total_angle          每位主人扫描的最大转动角（默认 400 度）
  ~search_start_text         开始寻找时的播报
  ~search_each_text          找某一位之前的播报（%s = 姓名）
  ~owner_action_start_text   找到某人后的播报（%s = 姓名）
  ~owner_missing_text        某一位没找到时的播报（%s = 姓名）
  ~all_done_text             全部结束的播报
  ~search_waypoints          要找的房间（逗号分隔，默认 living_room,kitchen,bedroom,canteen）
  ~exit_waypoint_name        回出口用的航点名（默认 exit）
  ~navigate_enabled          是否导航找人（默认 true；false = 原地转圈）
  ~navigate_dry_run          空跑：不真的开车，直接当作已到达每个航点（默认 false）
  ~waypoint_goal_timeout     单个航点的导航预算秒数，含重试（默认 45）
  ~navigation_failure_limit  同一目标（每间房/起点/回出口）连续失败几次就
                             放弃该目标的导航（默认 3；下一间重新计数继续导航）
  ~min_room_scan_seconds     一间房至少留多少秒才值得进去扫（默认 40）
  ~competition_total_seconds 比赛总时长，用来算时间预算（默认 480 = 8 分钟）
  ~competition_exit_reserve_seconds  给回出口预留的秒数（默认 60）

走近与对话（播报后的实际反应）
------------------------------
  ~approach_enabled           是否走近主人（默认 true）
  ~approach_standoff_distance 停到离主人多少米（默认 1.0，用点云量真距离）
  ~approach_min_distance      安全下限，比这个还近就不再往前（默认 0.6）
  ~approach_pointcloud_enabled 是否用点云量距离（默认 true；关掉就退回框高占比）
  ~approach_standoff_ratio    点云不可用时的退路：框高占比（默认 0.85，不精确）
  ~approach_camera_pitch_deg  相机俯仰角（默认 18°）。点云 z 是沿光轴的距离，
                              按这个角投影到水平面再取分位数 —— 不修正的话
                              "躺在地上的人"会被多算 0.2~0.3 米
  ~approach_target_min_iou    靠近时"还是同一个人"的判定：与上一帧目标框的
                              IoU 下限（默认 0.15），低于它就不跟（防止被路人抢目标）
  ~approach_forward_speed     走近时的前进速度 m/s（默认 0.15）
  ~approach_max_seconds       走近最多花多少秒（默认 20）

  挥手求助的"走近"从 2026-10-16 起改成**导航式**（和摔倒靠近同一套路），
  原因见 approach_owner_navigate 的注释：老的 approach_owner 是纯相机闭环 + 直发
  /cmd_vel，既不查 /scan 也不走代价地图，主人站在椅子后面挥手时会撞椅子腿。
  ~approach_navigate_enabled   走近是否先用 move_base 走到 standoff 点（默认 true）
                               关掉就退回老行为（纯相机闭环，**没有雷达护栏**）
  ~approach_nav_stage_standoff_distance 中间点离主人多少米（默认 0 = 用
                               approach_standoff_distance，即 1.0 m）
  ~approach_nav_timeout        导航段最多多少秒（默认 25）
  ~approach_nav_obstacle_retries 中间点被挡时换一侧重试几次（默认 1）
  ~approach_nav_obstacle_shift 每次换边横移多少米（默认 0.35）
  ~approach_nav_fallback_camera 导航没走通时是否退回相机闭环（默认 false =
                               原地不动直接进入问答；true = 退回老办法，风险自负）
  ~approach_face_first_enabled 导航前是否先原地"裸转正"（默认 false：裸转没有
                               护栏，交给 move_base 自己转目标朝向更安全）
  ~approach_face_first_max_angle 偏多少度以上才考虑转正（默认 0.35 rad）
  ~approach_creep_max_distance 收尾允许用里程计"盲走"补齐的最大距离
                               （**默认 0 = 不盲走**；站姿主人点云量得到，
                                没必要盲贴。要给也只能给 0.1~0.2 兜底）
  ~approach_finish_retries     收尾点云闭环最多再补几次（默认 1）
  ~approach_owner_band_low/high 点云取"身体带"的高度比例（默认 0.15 / 0.85）
  ~ask_help_enabled           走近后是否问"请问您需要什么帮助"（默认 true）
  ~ask_help_retries           没听清时重问几轮（默认 1，即最多问 2 轮）
  ~ask_help_answer_timeout    每轮等主人回答的秒数（默认 8）
  ~listen_settle_seconds      收到第一条识别结果后再多等一窗，取更完整的那版（默认 2.2）

雷达护栏（三条，全部可关）
--------------------------
  ~fall_scan_guard_distance   直行/爬行段的**正前方**硬护栏，米（默认 0.40）
  ~scan_forward_half_angle_deg 上面那条护栏的角度窗口，±度（默认 25）
  ~scan_turn_guard_distance   **原地旋转护栏**：全向（360°）最近距离 ≤ 这个值就
                              暂停旋转/停止转正（默认 0.30；0=关闭）。
                              车体旋转会横扫侧面的椅子腿，正前方窗口管不到，
                              所以这条必须看全向。低于车体半径(0.25)没意义；
                              调太大时，机器人停在离墙 0.3 m 处会把该房间的
                              搜索卡住（日志/终端会打「旋转护栏」）。
                              生效范围：转圈找人（scan_for_owner_for_target）、
                              转正（rotate_toward_bearing）。

  挥手对话的三句话（规则 2.6-5 判据是"按原意复述"）：
    问：  请问您需要什么帮助？
    重问：我没有听清，请您再说一遍需要什么帮助。
    复述：您需要的是：<主人的原话>。我明白了，马上为您处理。

电器开关（只播报，不动机械臂）
------------------------------
  ~ask_switch_enabled         是否问"要不要操作电器开关"（默认 true）
  ~ask_switch_answer_timeout  等主人回答的秒数（默认 8，只问一轮）
  开关标记的视觉检测 + 机械臂对位：硬件暂时不支持，留空位 ——
  operate_switch() 现在只播报状态，一条机械臂指令都不发。

摔倒协助（走近 + 伸机械臂）
--------------------------
  ~fall_assist_enabled             是否做摔倒协助（默认 true）
  ~fall_approach_standoff_distance 摔倒时走到多少米（默认 **0.5**，赛场要求；
                                   Kinect v2 深度下限也是 ~0.5 米，是最脆的一处）
  ~fall_approach_min_distance      摔倒时的安全下限（默认 0.35；
                                   它比通用的 ~approach_min_distance(0.6) 更低，
                                   否则 0.6 米的目标会被通用下限挡在 0.6 米）
  ~fall_assist_arm_enabled         是否真的动机械臂（默认 true）
  ~mani_ctrl_topic                 机械臂话题（默认 /wpb_home/mani_ctrl）
  ~fall_assist_arm_extend_lift     伸出去的 lift（默认 0.50）
  ~fall_assist_arm_hold_seconds    伸出去保持几秒（默认 4.0）
  ~fall_assist_arm_retract_lift    收回的 lift（默认 0.0）
  数值取自旧包（同一台机器上实测用过的一套）。指令是"持续发"的：
  机械臂控制器只认连续收到的目标，发一次就停它可能在半路停住。

坐姿/躺姿识别（人脸与身形怎么配合）
----------------------------------
  ~face_accept_threshold          人脸相似度接受线（默认 0.45）
  ~sitting_appearance_threshold   坐着且看不到脸时，外观分数要求（默认 0.75）
  ~sitting_ignore_small_face_px   人脸框小于这么多像素就不采信它（默认 0 = 关）
  背景：2~3 米外的人脸只有三四十像素，同一段视频里相似度能在 0.09~0.44 乱跳，
  却能一票否决掉 0.77 的身形匹配。打开这个开关会放宽这一点，但也会放进
  "坐着的陌生人"（陌生人实测身形分能到 0.9 上下），所以默认关着。

地面垃圾（只找 + 播报，不抓取）
------------------------------
  ~trash_announce_enabled     是否做垃圾识别与播报（默认 true）
  ~trash_detections_topic     垃圾检测话题（留空 = 和主人检测同一个话题）
  ~trash_stability_frames     窗口内要命中几次才算数（默认 4）
  ~trash_observation_window_seconds  滑动窗口长度（默认 10 秒）
  ~trash_min_score            分数阈值（默认 0.30）
  ~trash_min_score_by_kind    按类单独设阈值，如 "纸团:0.18"
  ~trash_min_box_pixels       框长边像素下限（默认 16，**看长边不看短边**）
  ~trash_min_center_y_ratio   中心必须在画面下方这个比例之下（默认 0.55，挡桌上的东西）
  ~trash_found_template       播报模板（默认 "我找到了垃圾：%s。"）
  检测来自独立的 YOLO-World 节点（launch 配置 ⑭）；词表里的英文提示词
  必须都能在 TRASH_CLASS_NAMES 里查到中文名，否则会被静默忽略。

时间预算怎么算
--------------
8 分钟里注册就占掉约 2 分钟，**注册也是算在比赛时间里的**，所以计时从 run()
开头（注册之前）就开始。剩下的每个房间只可能是"导航十几秒 + 转一圈约 40 秒"，
所以：
  · 剩余时间 < ~competition_exit_reserve_seconds  → 一个房间都不再进，直接收尾；
  · 剩余时间 - 预留 < ~min_room_scan_seconds      → 这一间不进去扫，直接收尾；
  · 单次导航最多等 min(~waypoint_goal_timeout, 剩余时间 - 预留)，重试也在这份预算里。
任何一条都不会让"走回出口"的时间被吃掉。
"""

import importlib.util
import math
import os
import sys
import threading
import time

import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Twist
from move_base_msgs.msg import MoveBaseGoal
from perception_msgs.msg import Detection2DArray
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Bool

# 摔倒靠近的调试可视化（RViz Marker）。用 try 包住：这是纯调试功能，
# 万一某台机器没装 visualization_msgs，也绝不能把任务节点带崩。
try:  # pragma: no cover - 取决于运行环境
    from visualization_msgs.msg import Marker, MarkerArray
except Exception:  # noqa: BLE001
    Marker = None
    MarkerArray = None

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from face_body_enroll_test import FaceBodyEnrollTest  # noqa: E402


# ----------------------------------------------------------------------
# 地面垃圾识别（"找到垃圾"那 60 分）
# ----------------------------------------------------------------------
# 这部分是从 cpy 的工作区（catkin_ws1）里搬过来的，数值和判据顺序保持原样，
# 只去掉了台架探针和"捡起/入桶"那两段（那两段我们做不了，也不够分）。
#
# 检测来源是 **YOLO-World 开放词表节点**：英文提示词直接成为检测框的
# class_name，这里把它翻成中文用于播报。
# 规则 2.5 赛前准备 2：现场三种垃圾是「空瓶子、纸团、水果皮」。
#
# **每加一个提示词都必须在这里补一行中英映射**，否则 YOLO 会照常输出那个类名，
# 而下面会把它当"非垃圾类"静默忽略 —— 看起来像"模型认不出"，实际是配置漏了。
TRASH_CLASS_NAMES = {
    "bottle": "空瓶子",
    "empty bottle": "空瓶子",
    "water bottle": "空瓶子",
    "plastic bottle": "空瓶子",
    "drinking bottle": "空瓶子",
    "can": "易拉罐",
    "soda can": "易拉罐",
    "crumpled paper": "纸团",
    "paper ball": "纸团",
    "paper wad": "纸团",
    "ball of paper": "纸团",
    "waste paper": "纸团",
    "paper lump": "纸团",
    "tissue": "纸团",
    "paper cup": "纸杯",
    "banana peel": "水果皮",
    "fruit peel": "水果皮",
    "orange peel": "水果皮",
    "apple peel": "水果皮",
    "fruit skin": "水果皮",
    "apple core": "水果皮",
    "plastic bag": "塑料袋",
    "trash": "垃圾",
    "garbage": "垃圾",
}

# 判据默认值（取自 cpy 现场实测：一轮 2861 帧 / 468s 的命中率）
TRASH_DEFAULT_MIN_SCORE = 0.30
TRASH_DEFAULT_MIN_BOX_PIXELS = 16
TRASH_MIN_BOX_SHORT_SIDE = 3
TRASH_DEFAULT_MIN_CENTER_Y_RATIO = 0.55
TRASH_DEFAULT_STABILITY_FRAMES = 4
TRASH_DEFAULT_WINDOW_SECONDS = 10.0

# 找到垃圾时说的话。规则 2.6-6 的判据是"以语音播报为主要依据"。
TRASH_FOUND_TEMPLATE = "我找到了垃圾：%s。"


class AllOwnersTask(FaceBodyEnrollTest):
    """注册 N 位主人，再逐个找到他们并分别播报姓名与动作。"""

    def __init__(self):
        super().__init__()
        # 每位主人单独扫一整圈：400° 比 360° 多 40°，给起止接缝留重叠，
        # 免得人正好站在"起点=终点"那个缝里被漏掉。
        self.scan_total_angle = max(
            0.1, float(rospy.get_param("~scan_total_angle", math.radians(400.0)))
        )
        self.scan_timeout = max(1.0, float(rospy.get_param("~scan_timeout", 90.0)))

        # 导航开关沿用父类已有的 ~navigate_enabled，这里默认 true：
        # 新包里已经有真实地图（maps/map.yaml）和 5 个航点（~/waypoints.xml），
        # 所以默认就"逐房间导航找人"。想退回原地转圈（调试识别算法、
        # 或者现场没建图/定位不可用）就传 _navigate_enabled:=false。
        self.navigate_enabled = bool(
            rospy.get_param("~navigate_enabled", True)
        )

        # ---- 导航参数（跨房间找人）----
        # 要找的房间列表，按顺序去。名字必须和航点文件里的 <Name> 完全一致。
        # 位置越靠前的房间越先去——把"主人最可能在的房间"排前面最省时间。
        self.search_waypoints = self.parse_name_list(
            rospy.get_param(
                "~search_waypoints",
                "living_room,kitchen,bedroom,canteen",
            )
        )
        # 全部找完之后回出口用的航点名；留空 = 不回出口
        self.exit_waypoint_name = str(
            rospy.get_param("~exit_waypoint_name", "exit")
        ).strip()
        self.return_to_exit_enabled = bool(
            rospy.get_param("~return_to_exit_enabled", True)
        )
        # 空跑模式：不真的开车，但流程上"当作已经到达"这个航点 —— 说完
        # "正在前往X"就直接进入这个房间的扫描找人。
        # 什么时候用：场地/地图对不上、或者只想在当前位置测"到了之后那一套"
        # （转圈找人 → 居中 → 报姓名 → 动作 → 反应）时，不用为了测这一段
        # 先把车开到客厅去。配合 ~search_waypoints 只留一个房间最省时间。
        self.navigate_dry_run = bool(rospy.get_param("~navigate_dry_run", False))
        # 单个航点的导航预算：**包含失败后重试一次**，不是每次各给这么多。
        # 父类默认 120 秒，对 8 分钟的比赛太长 —— 房间之间走一趟一般十几二十秒。
        # 这个值还同时受"剩余时间"约束（见 navigation_timeout_for）：剩余时间
        # 只够 20 秒就只等 20 秒，绝不会因为一次导航把预算烧穿、走不回出口。
        # ⚠️ 这个值要等实测：等真实地图上跑通一次导航后，用日志里
        #    「前往航点」到「到达航点」之间的秒数再定，别照搬 45。
        self.waypoint_goal_timeout = max(
            5.0, float(rospy.get_param("~waypoint_goal_timeout", 45.0))
        )
        # 剩余时间少于这个数就不值得再导航了（走过去 + 找人都不够）
        self.min_navigation_seconds = max(
            0.0, float(rospy.get_param("~min_navigation_seconds", 12.0))
        )
        # 导航失败后"清一次代价地图再试一次"。这是 move_base 最常见的自愈手段：
        # 上一次失败可能只是局部代价地图里堆了障碍（人走过、激光噪点），
        # 清掉重发目标往往就好了。
        self.retry_navigation_after_clear = bool(
            rospy.get_param("~retry_navigation_after_clear", True)
        )
        # 同一间房连续失败到这个次数，就**本间**放弃导航（成绩照旧保住），
        # 但下一间房会重新计数、照常导航。
        # 2026-10-16 现场改了语义：原来是"整场"计数 —— 前两间失败后整场都停用
        # 导航，后面所有房间都变成"原地转圈"，现场看起来像车在乱跑
        # （日志：导航连续失败 2 次…剩下的房间只做原地识别）。
        # 现在按"目标"计数（每间房 / 起点 / 回出口各自清零），阈值也从 2 提到 3。
        self.navigation_failure_limit = max(
            1, int(rospy.get_param("~navigation_failure_limit", 3))
        )
        self.navigation_failure_count = 0
        # 整场累计失败次数（只用于日志/复盘，不影响策略）
        self.navigation_failure_total = 0
        # ---- 开机硬件自检（2026-10-16 加，移植自 server_robot3 的
        #      wait_for_hardware_inputs）--------------------------------------
        # 为什么必须有：实测出现过"雷达没插 / ttyUSB 掉了"还照常起任务、
        # 一路开出去的情况 —— AMCL 收不到激光就不再更新，代价地图也标不出障碍，
        # 最后表现成"导航莫名其妙失败"，现场很难一眼看出根因。现在起任务先等
        # /scan 和 /odom 真的出数据（相机已经在 wait_for_camera_inputs 里单独把关），
        # 缺哪个就明确报出来并停车，不再盲开。
        # 调试用（不想等/没接底盘）：_require_scan:=false _require_odom:=false
        self.require_scan = bool(rospy.get_param("~require_scan", True))
        self.require_odom = bool(rospy.get_param("~require_odom", True))
        self.hardware_check_timeout = max(
            1.0, float(rospy.get_param("~hardware_check_timeout", 30.0))
        )
        # 航点名 → 播报用的中文名（说话时念"客厅"比念"living_room"自然）
        self.waypoint_speech_names = {
            "living_room": "客厅",
            "kitchen": "厨房",
            "bedroom": "卧室",
            "canteen": "餐厅",
            "exit": "出口",
        }
        self.search_go_text = rospy.get_param(
            "~search_go_text", "正在前往%s寻找主人。"
        )
        # 备用航点（2026-10-15 现场要求）：形如 "bedroom2=bedroom"。
        # 只有主航点那一间一个人都没找到时才去备用航点；主航点找到人就跳过它。
        # _search_waypoints 里要把备用航点写在主航点后面，例如
        #   living_room,kitchen,canteen,bedroom,bedroom2
        self.fallback_waypoint_anchors = self.parse_fallback_waypoints(
            rospy.get_param("~fallback_waypoints", "")
        )
        # 赛制（2026-10-16 确认）：**每个房间至多只有一位主人**。
        # 所以"本间认到一位主人"就立刻结束这一间的扫描，直接去下一个航点，
        # 不再为"同房间的第二位主人"多转一圈（一圈约 40 秒，纯浪费）。
        self.one_owner_per_room = bool(
            rospy.get_param("~one_owner_per_room", True)
        )
        # ---- 开局自动给 AMCL 一个初始位姿（2026-10-16）----
        # 机器人按约定停在某个航点（默认 = 回出口的 exit，也就是"门口"），
        # 开机直接把该航点的位姿发到 /initialpose，AMCL 立刻收敛 ——
        # 省掉"每次起栈都去 RViz 点一次 2D Pose Estimate"。
        # 机器人没停在那儿、或想自己微调：_auto_initial_pose:=false 回到手动点。
        self.auto_initial_pose = bool(rospy.get_param("~auto_initial_pose", True))
        self.auto_initial_pose_waypoint = str(
            rospy.get_param("~auto_initial_pose_waypoint", "") or ""
        ).strip()
        self.auto_initial_pose_wait = max(
            0.0, float(rospy.get_param("~auto_initial_pose_wait", 2.0))
        )
        self._initial_pose_pub = None
        # 当前所在房间（航点名）：动作播报里要念出来（主人张三正在客厅坐着）。
        # 由 run_waypoint_search_round 每进一间就更新。
        self.current_room_waypoint = ""

        # ---- 比赛时间预算 ----
        # 8 分钟（480 秒）是规则给的时限。留 ~competition_exit_reserve_seconds
        # 秒用于"回出口"和收尾，到点就不再接新房间。计时从 run() 开头算起，
        # 注册 + 找人 + 动作识别都在里面。
        self.competition_total_seconds = max(
            30.0, float(rospy.get_param("~competition_total_seconds", 480.0))
        )
        self.competition_exit_reserve_seconds = max(
            0.0, float(rospy.get_param("~competition_exit_reserve_seconds", 60.0))
        )
        # 一间房至少要留这么多秒才值得进去扫。
        # 一圈实测约 40 秒（400 度 / 0.25 rad/s + 每步 0.35 秒停顿），
        # 少于这个数进去也是白转，不如把时间留给"走回出口"。
        self.min_room_scan_seconds = max(
            1.0, float(rospy.get_param("~min_room_scan_seconds", 40.0))
        )
        self.competition_started_time = None
        # ---- 时间预算开关（2026-10-16 现场要求）----
        # 默认 **false = 不再用"已用时长"裁剪流程**：
        #   · 不再因为"剩余时间不够"跳过一个房间；
        #   · 不再因为剩余时间缩短单航点导航超时/单间扫描时长；
        #   · 不再打印"比赛时间预算 | …"那些行；
        # 各操作的固定超时（单航点 45s、单间扫描 90s、等）**保留**，
        # 那是防卡死的安全边界，不是"计时统计"。
        # 想恢复"8 分钟限时里自动取舍"：起栈传 _competition_time_budget_enabled:=true。
        self.competition_time_budget_enabled = bool(
            rospy.get_param("~competition_time_budget_enabled", False)
        )

        # ---- 地面垃圾识别（"找到垃圾"60 分）----
        # 捡起 + 入桶做不了（没有抓取能力），但"找到垃圾并说出名字"本身就
        # 按规则 2.6-6 计分，判据就是语音播报。
        #
        # 检测是零额外耗时的：机器人找主人 / 居中 / 认动作时相机本来就在扫这个
        # 房间，检测框从 /perception/person_detections_2d 一路流过来，我们只是
        # 顺手记一笔；等这个房间的流程走完再看累积结果决定要不要播报。
        # ⚠️ 默认关闭：找垃圾要靠 YOLO-World 开放词表检测器，实测在本机（16 核）
        # 上即便压到 2 线程 + imgsz480，每 3 帧推理一次也要持续吃掉 1 个多核，
        # 而主任务的姿态/ReID/人脸都在同一个 CPU 上 —— 设备和 GPU 都撑不住，
        # 所以这一部分分数**放弃**。代码留着（判据、累积、播报都是好的，也测过），
        # 将来换了机器把 ~trash_announce_enabled 打开、并把检测话题
        # ~trash_detections_topic 指到一个 YOLO-World 节点的输出即可。
        self.trash_announce_enabled = bool(
            rospy.get_param("~trash_announce_enabled", False)
        )
        self.trash_class_names = dict(TRASH_CLASS_NAMES)
        self.trash_found_template = rospy.get_param(
            "~trash_found_template", TRASH_FOUND_TEMPLATE
        )
        self.trash_min_score = float(
            rospy.get_param("~trash_min_score", TRASH_DEFAULT_MIN_SCORE)
        )
        # 某类垃圾单独的分数阈值，例如 "纸团:0.18"。
        # 为什么需要：纸团在实测里最高分只有 0.21，用全局 0.30 会一个都播不出去；
        # 而瓶子和水果皮能到 0.6~0.9，阈值低了对它们反而是噪声。
        self.trash_min_score_by_kind = self.parse_float_map(
            rospy.get_param("~trash_min_score_by_kind", {})
        )
        self.trash_min_box_pixels = max(
            1.0, float(rospy.get_param("~trash_min_box_pixels", TRASH_DEFAULT_MIN_BOX_PIXELS))
        )
        self.trash_min_center_y_ratio = max(
            0.0,
            min(
                1.0,
                float(
                    rospy.get_param(
                        "~trash_min_center_y_ratio", TRASH_DEFAULT_MIN_CENTER_Y_RATIO
                    )
                ),
            ),
        )
        self.trash_stability_frames = max(
            1, int(rospy.get_param("~trash_stability_frames", TRASH_DEFAULT_STABILITY_FRAMES))
        )
        self.trash_observation_window_seconds = max(
            1.0,
            float(
                rospy.get_param(
                    "~trash_observation_window_seconds", TRASH_DEFAULT_WINDOW_SECONDS
                )
            ),
        )
        # 中文垃圾名 -> {"hits": [命中时刻...], "best_score": float, "label": 英文类名}
        self.trash_observations = {}
        # 已经播报过的中文名：同一类垃圾只说一次，别一路重复
        self.trash_announced = set()
        # YOLO 到底吐出过哪些类名（现场归因用：一次 0 命中是"没放垃圾"还是
        # "模型没认出来"，只看这个就能分辨）
        self.trash_class_histogram = {}
        self.trash_no_height_warned_at = 0.0
        # 垃圾检测走哪个话题。
        # 留空 = 跟主人检测同一个话题（cpy 就是这么做的：一个 YOLO-World 同时
        # 出 person 和垃圾类，各按 class_name 过滤）。填了就单独订一路，
        # 这样**完全不动现有的主人检测链路**，垃圾检测单独起停都能用。
        self.trash_detections_topic = str(
            rospy.get_param("~trash_detections_topic", "")
        ).strip()
        self.trash_detections = []
        self.trash_detections_stamp = None
        self.trash_detections_sub = None
        # 垃圾检测节点（launch ⑭）的 pause 话题。填上之后，主任务会在不需要
        # 看垃圾的阶段把它暂停 —— 因为它在 CPU 上推理很贵（见下面 ⑭ 的注释）：
        #   注册 2 分钟           -> 暂停（完全不需要垃圾）
        #   寻找主人（导航/扫房间）-> 恢复（垃圾就在这时候看）
        #   动作识别那 20~30 秒    -> 暂时暂停（CPU 最紧张，而且这时镜头对着主人）
        #   收尾/回出口            -> 暂停
        self.trash_node_pause_topic = str(
            rospy.get_param("~trash_node_pause_topic", "")
        ).strip()
        self.trash_pause_pub = None
        self.trash_detector_paused = None
        if (
            self.trash_detections_topic
            and self.trash_detections_topic != self.detections_topic
        ):
            try:
                self.trash_detections_sub = rospy.Subscriber(
                    self.trash_detections_topic,
                    Detection2DArray,
                    self.trash_detections_callback,
                    queue_size=1,
                )
                rospy.loginfo(
                    "垃圾检测单独订阅：%s（与主人检测的 %s 分开）",
                    self.trash_detections_topic,
                    self.detections_topic,
                )
            except Exception as exc:
                rospy.logwarn("订阅垃圾检测话题失败：%s", exc)
                self.trash_detections_sub = None

        # ---- 播报后的实际反应：走近主人 + 挥手求助对话 ----
        # 动作播报完之后该"做点什么"，对应规则里"根据行为/指令做出正确人机交互"
        # 那 150 分。挥手求助这一条规则的验收是"走近 + 语音问答，能把需求按原意
        # 复述"，所以这里就干两件事：走近、问一句、把主人的原话念回去。
        self.approach_enabled = bool(rospy.get_param("~approach_enabled", True))
        self.approach_start_text = rospy.get_param(
            "~approach_start_text", "好的，我这就过来。"
        )
        # 怎么算"到跟前了"：**用点云量真实距离**，停到
        # ~approach_standoff_distance 米（默认 1.0）。
        #
        # 为什么不用"人的框在画面里占多大"：那个换算成米要先知道相机高度和俯仰角，
        # 而这两个数在本机上有两个互相矛盾的口径（动作节点日志里同时出现过
        # "camera height 0.844m" 和由地面线反推的约 1.44m）。更麻烦的是相机是
        # 低头装的，站姿的人一近到 1.5 米左右头顶就出画、脚也快出画，
        # 框会被削平 —— 这时"占比"已经饱和，分不出 1.5 米还是 1.0 米。
        # 点云直接给深度，不需要任何标定，这是唯一能稳稳停到 1 米的办法。
        #
        # 代价（本机实测）：点云**不占显存**（纯 CPU 侧数据搬运，不建 CUDA 上下文），
        # 每帧 9.9MB 的反序列化约 0.5ms，读人体框内一小块约 0.23ms。
        self.approach_pointcloud_enabled = bool(
            rospy.get_param("~approach_pointcloud_enabled", True)
        )
        self.points_topic = rospy.get_param("~points_topic", "/kinect2/qhd/points")
        self.approach_pointcloud_max_age = max(
            0.1, float(rospy.get_param("~approach_pointcloud_max_age", 1.0))
        )
        # 目标距离：1.0 米 —— 太远裁判会以为"没靠近"，太近不安全也不好说话。
        self.approach_standoff_distance = max(
            0.3, float(rospy.get_param("~approach_standoff_distance", 1.0))
        )
        # 硬下限：比这个还近就绝不再往前发速度（安全底线）
        self.approach_min_distance = max(
            0.2,
            min(
                float(rospy.get_param("~approach_min_distance", 0.6)),
                self.approach_standoff_distance,
            ),
        )
        # 取框内点云深度的低分位数当"人的距离"：背景（墙）比人远，
        # 取靠前的分位数就不容易被背景带偏；再配合"只取框的上半部分"
        # 避开地面（低头相机在近处会把人身前的地面照进框里，那比人还近）。
        self.approach_depth_percentile = max(
            5.0, min(50.0, float(rospy.get_param("~approach_depth_percentile", 25.0)))
        )
        self.approach_depth_min_samples = max(
            5, int(rospy.get_param("~approach_depth_min_samples", 20))
        )
        # 点云不可用时的退路：还是用框高占比（只是粗略，日志里会写明是估计值）
        self.approach_standoff_ratio = max(
            0.35, min(1.0, float(rospy.get_param("~approach_standoff_ratio", 0.85)))
        )
        self.approach_forward_speed = max(
            0.05, min(0.35, float(rospy.get_param("~approach_forward_speed", 0.15)))
        )
        self.approach_turn_speed = max(
            0.10, min(0.80, abs(float(rospy.get_param("~approach_turn_speed", 0.35))))
        )
        self.approach_center_tolerance = max(
            0.02, float(rospy.get_param("~approach_center_tolerance", 0.10))
        )
        self.approach_max_seconds = max(
            3.0, float(rospy.get_param("~approach_max_seconds", 20.0))
        )
        # 连续多久看不到主人就停下 —— 绝不"盲开"：人一走开就停。
        self.approach_lost_seconds = max(
            0.5, float(rospy.get_param("~approach_lost_seconds", 2.0))
        )
        # ---- 挥手求助"走近"改成导航式（2026-10-16）----
        # 现场问题：开了雷达和硬护栏，挥手靠近时还是撞到椅子腿。原因不是护栏参数，
        # 而是**这条路上根本没有护栏**：老的 approach_owner 是纯相机闭环 + 直发
        # /cmd_vel，不查 /scan、也不走代价地图，雷达把椅子腿标进 costmap 对它没约束。
        # 现在与摔倒靠近一致：点云快照出主人的 map 系位置 → 退 standoff 得到中间点
        # → 交 move_base（走代价地图绕障，导航段带 /scan 硬护栏）→ 最后一段用点云
        # 闭环（同样带 /scan 护栏）收到 approach_standoff_distance。
        self.approach_navigate_enabled = bool(
            rospy.get_param("~approach_navigate_enabled", True)
        )
        # 0 = 用 approach_standoff_distance 当中间点距离
        self.approach_nav_stage_standoff_distance = max(
            0.0, float(rospy.get_param("~approach_nav_stage_standoff_distance", 0.0))
        )
        self.approach_nav_timeout = max(
            5.0, float(rospy.get_param("~approach_nav_timeout", 25.0))
        )
        self.approach_nav_obstacle_retries = max(
            0, int(rospy.get_param("~approach_nav_obstacle_retries", 1))
        )
        self.approach_nav_obstacle_shift = max(
            0.10, float(rospy.get_param("~approach_nav_obstacle_shift", 0.35))
        )
        # 导航没走通时：默认**原地不动**直接进入问答，不退回"没有护栏的相机闭环"。
        self.approach_navigate_fallback_camera = bool(
            rospy.get_param("~approach_navigate_fallback_camera", False)
        )
        self.approach_face_first_enabled = bool(
            rospy.get_param("~approach_face_first_enabled", False)
        )
        self.approach_face_first_max_angle = max(
            0.05, float(rospy.get_param("~approach_face_first_max_angle", 0.35))
        )
        # 挥手靠近最后**不盲走**（2026-10-16 现场要求）：站姿主人 1 m 内点云量得到，
        # 没必要用纯里程计往前贴。默认 0 = 关闭；真需要兜底再手动给个 0.1~0.2。
        self.approach_creep_max_distance = max(
            0.0, float(rospy.get_param("~approach_creep_max_distance", 0.0))
        )
        # 收尾时点云闭环补几次（默认 1 = 最多再补一次，全程都有点云+护栏确认）
        self.approach_finish_retries = max(
            0, int(rospy.get_param("~approach_finish_retries", 1))
        )
        self.approach_owner_band_low = max(
            0.0, min(0.80, float(rospy.get_param("~approach_owner_band_low", 0.15)))
        )
        self.approach_owner_band_high = max(
            self.approach_owner_band_low + 0.05,
            min(1.0, float(rospy.get_param("~approach_owner_band_high", 0.85))),
        )
        # 点云订阅是"要走近时才建"的（懒加载），没走近就不订，省一份订阅
        self.approach_pointcloud_sub = None
        self.approach_pointcloud = None
        self.approach_pointcloud_stamp = None
        self.approach_pointcloud_warned = False
        # 相机俯仰角：点云是相机光学系（x 右 / y 下 / z 前），相机是**向下俯**装的，
        # 所以 z 是"沿光轴"的距离而不是水平距离。对躺在地上的人尤其明显：
        # 实测 2026-10-09 21:24（摔倒协助）：人几乎贴到跟前，读数还停在 1.25~1.47 米。
        # 默认 18°（与动作节点自动标定出的 17.8° 一致），按它把点投影到水平面。
        self.approach_camera_pitch_deg = max(
            0.0, min(60.0, float(rospy.get_param("~approach_camera_pitch_deg", 18.0)))
        )
        # ---- 靠近时的"身份/连续性闸门" ------------------------------------
        # 以前选目标只按"离上次的中心最近"，画面里走过别人时目标会被抢走
        # （21:24 那次日志里距离从 2.04 突然跳到 4.21 米就是这个味道）。
        # 现在要求候选框与上一帧的目标框有足够重叠（IoU）才算"还是同一个人"。
        # ⚠️ 2026-10-10 现场决定：**摔倒协助那条靠近路径不做身份校验**（房间里不会同时
        #   有两位主人，而且躺姿框在"宽扁/窄长"之间抖动时 IoU≈0，严格闸门反而把真主人
        #   挡掉——那次就停在 1.58 米没到位）。摔倒分支单独传 identity_gate=False 关掉它；
        #   挥手求助那条路径保持本参数的默认值（True = 原来的严格跟踪）。
        self.approach_identity_gate = bool(
            rospy.get_param("~approach_identity_gate", True)
        )
        self.approach_target_min_iou = max(
            0.0, min(0.9, float(rospy.get_param("~approach_target_min_iou", 0.15)))
        )
        # 单帧漏检时的宽容：上一帧目标 0.x 秒内还看得见、且新候选离上次中心很近，
        # 才允许"续上"；超过这个时间/位置就当目标丢了，原地等，绝不去跟别人。
        self.approach_relock_center_tolerance = max(
            0.0,
            min(0.5, float(rospy.get_param("~approach_relock_center_tolerance", 0.10))),
        )
        self.approach_relock_grace_seconds = max(
            0.0, float(rospy.get_param("~approach_relock_grace_seconds", 0.6))
        )

        # 挥手求助：问一句 → 听回答 → 把原话复述回去
        self.ask_help_enabled = bool(rospy.get_param("~ask_help_enabled", True))
        # 问句用规则里的原话「咨询主人需要什么帮助」——
        # 旧包和 cpy 那份用的都是这句，判分时更容易被认出来。
        self.ask_help_question_text = rospy.get_param(
            "~ask_help_question_text", "请问您需要什么帮助？"
        )
        # 复述必须用主人自己的话（%s 就是 ASR 原话），不能自己改写成
        # "好的我这就去开灯" —— 规则 2.6-5 要的是"按原意复述"。
        # 这句是 cpy 那份里对着规则写好的措辞，直接沿用。
        self.ask_help_retell_text = rospy.get_param(
            "~ask_help_retell_text", "您需要的是：%s。我明白了，马上为您处理。"
        )
        # 没听清时再问一遍（旧包那句现成的话，直接沿用）
        self.ask_help_retry_text = rospy.get_param(
            "~ask_help_retry_text", "我没有听清，请您再说一遍需要什么帮助。"
        )
        self.ask_help_timeout_text = rospy.get_param(
            "~ask_help_timeout_text", "抱歉，我没有听清您的需求。"
        )
        # 问几轮。cpy 那份是"问一次 + 重问一次"（共 2 轮）——这 50 分就在
        # "真的听见并复述出来"上，多问一轮值；再多就把 8 分钟耗掉了。
        self.ask_help_retries = max(
            0, int(rospy.get_param("~ask_help_retries", 1))
        )
        self.ask_help_answer_timeout = max(
            3.0, float(rospy.get_param("~ask_help_answer_timeout", 8.0))
        )
        # 收到第一条识别结果后再多等一窗。
        # ASR 是重叠窗口发布的（约每 2 秒一条，先短后长："帮我开" → "帮我开一下灯"），
        # 拿到第一条就走，复述出来的是被切掉半截的话。这一等是为了收完整。
        self.listen_settle_seconds = max(
            0.0, float(rospy.get_param("~listen_settle_seconds", 2.2))
        )

        # ---- 播报后的实际反应（三）：电器开关 ----
        # 硬件暂时不支持"开关标记视觉检测 + 机械臂对位"，所以这一段只做到
        # "听懂 + 播报"：问一句 → 从主人的话里听出要开还是要关 → 把开关的状态
        # 用话说清楚。机械臂那半留空位（见 operate_switch 的说明）。
        self.ask_switch_enabled = bool(rospy.get_param("~ask_switch_enabled", True))
        self.ask_switch_question_text = rospy.get_param(
            "~ask_switch_question_text",
            "需要我帮您操作电器开关吗？要打开还是关闭？",
        )
        self.ask_switch_no_answer_text = rospy.get_param(
            "~ask_switch_no_answer_text",
            "没有听清，那我先不动电器开关了。",
        )
        # 只问一轮。问两轮能把识别率提上去，但每个主人多花十几秒 ——
        # 8 分钟的比赛里这笔账不划算；真不够用把这里调大即可。
        self.ask_switch_answer_timeout = max(
            3.0, float(rospy.get_param("~ask_switch_answer_timeout", 8.0))
        )

        # ---- 播报后的实际反应（四）：摔倒协助（走近 + 伸机械臂）----
        # 规则场景②的验收是"把机械臂伸到人体正上方"，所以这一段要做两件事：
        #   走过去（用点云量距离，躺在地上的人也能量准）
        #   伸机械臂（往 /wpb_home/mani_ctrl 发 lift/gripper 的 JointState）
        self.fall_assist_enabled = bool(rospy.get_param("~fall_assist_enabled", True))
        # 摔倒时要比"挥手对话"贴得更近，机械臂才够得着人。
        # 0.5 米 —— 注意它比通用的安全下限 ~approach_min_distance(0.6) 还近，
        # 所以摔倒这条路单独带一个更低的下限，否则会被通用下限挡在 0.6 米。
        # 摔倒协助要走到多近：**0.5 米**（赛场要求，2026-10-09 试过 0.6 米，太远）。
        # 注意 0.5 米压在 Kinect v2 的深度下限（~0.5 m）附近，是这套参数里最脆弱的一处；
        # 不过距离改成"水平距离"之后好多了：躺姿时人在水平 0.5 米处，相机看到躯干的
        # z ≈ 0.5·cos18° + 0.7·sin18° ≈ 0.69 米，仍在有效范围内（换成旧算法量的是
        # z 方向小分位，反而容易抓到背景点）。若日志再出现"量不到"，下一步改取样
        # 窗口/分位数（见交接文档第 11 节的建议 3），而不是再动这个距离。
        self.fall_approach_standoff_distance = max(
            0.3, float(rospy.get_param("~fall_approach_standoff_distance", 0.5))
        )
        self.fall_approach_min_distance = max(
            0.2,
            min(
                float(rospy.get_param("~fall_approach_min_distance", 0.35)),
                self.fall_approach_standoff_distance,
            ),
        )
        # 摔倒时"连续看不到人多长时间才放弃"。躺姿的人体框在画面里会闪断
        # （2026-10-10 现场：2 秒就中止，最后停在 1.58 米），所以单独给更长的时间。
        self.fall_approach_lost_seconds = max(
            1.0, float(rospy.get_param("~fall_approach_lost_seconds", 5.0))
        )
        # ---- 摔倒靠近：快照式（方案 B 第一步）-----------------------------
        # 为什么另开一条路径：相机装在约 1.46 m 高、下倾 17.8°，人躺在地上一旦近到
        # 水平 1.0~1.2 m 以内，躯干就落到画面下边缘之外（几何必然），姿态检测必然丢；
        # 而"边看边走"的闭环一旦丢人就停（实测停在 1.58 m）。参考实现（server_robot3）
        # 的做法是：先在远处"快照"主人的 3D 位置 → 朝它转正 → 直行到云快看不见 →
        # 最后用里程计爬完剩下的距离。
        #   camera   = 原来的"看着人闭环"（默认，行为不变）
        #   snapshot = 快照式（本步新加，先用 /cmd_vel，下一步再接 move_base 避障）
        self.fall_approach_mode = str(
            rospy.get_param("~fall_approach_mode", "camera")
        ).strip().lower()
        self.fall_snapshot_seconds = max(
            0.2, float(rospy.get_param("~fall_snapshot_seconds", 0.9))
        )
        self.fall_snapshot_min_samples = max(
            1, int(rospy.get_param("~fall_snapshot_min_samples", 2))
        )
        self.fall_snapshot_min_points = max(
            5, int(rospy.get_param("~fall_snapshot_min_points", 20))
        )
        # 先转正：|侧向角| 小于这个就算对准（rad）
        self.fall_snapshot_bearing_tolerance = max(
            0.02, float(rospy.get_param("~fall_snapshot_bearing_tolerance", 0.10))
        )
        self.fall_snapshot_turn_speed = max(
            0.10, min(0.60, float(rospy.get_param("~fall_snapshot_turn_speed", 0.35)))
        )
        self.fall_snapshot_align_timeout = max(
            1.0, float(rospy.get_param("~fall_snapshot_align_timeout", 8.0))
        )
        # 直行段
        self.fall_snapshot_speed = max(
            0.05, min(0.30, float(rospy.get_param("~fall_snapshot_speed", 0.14)))
        )
        self.fall_snapshot_drive_timeout = max(
            2.0, float(rospy.get_param("~fall_snapshot_drive_timeout", 25.0))
        )
        # 云测到这个距离就切"里程计爬行"（人还在画面里，但马上就会掉出去）
        self.fall_cloud_stop_distance = max(
            0.6, float(rospy.get_param("~fall_cloud_stop_distance", 1.20))
        )
        # 爬行段（纯里程计 + /scan 前向护栏）
        self.fall_creep_speed = max(
            0.03, min(0.15, float(rospy.get_param("~fall_creep_speed", 0.08)))
        )
        self.fall_creep_max_distance = max(
            0.0, float(rospy.get_param("~fall_creep_max_distance", 0.90))
        )
        self.fall_creep_finish_tolerance = max(
            0.02, float(rospy.get_param("~fall_creep_finish_tolerance", 0.06))
        )
        self.fall_creep_timeout = max(
            1.0, float(rospy.get_param("~fall_creep_timeout", 8.0))
        )
        # /scan 前向硬护栏：爬行时正前方小于这个距离就立刻停（人本身也会被雷达看到，
        # 所以这个值要比 0.5 m 目标更小一点，只当"撞上东西了"的兜底）
        # 2026-10-16 按用户要求 0.32 → 0.40：跟导航段的 _fall_nav_guard_distance
        # 统一，留出更多安全余量（桌子/箱子的边沿、雷达噪点都更容易被挡住）。
        self.fall_scan_guard_distance = max(
            0.20, float(rospy.get_param("~fall_scan_guard_distance", 0.40))
        )
        self.scan_topic = str(rospy.get_param("~scan_topic", "/scan"))
        self.scan_forward_half_angle_deg = max(
            5.0, float(rospy.get_param("~scan_forward_half_angle_deg", 25.0))
        )
        self._scan_sub = None
        self._scan_forward_min = None
        # 全向（360°）最近距离：原地旋转的护栏用它（正前方那条 25° 窗管不到侧面）
        self._scan_any_min = None
        self._scan_stamp = 0.0
        # 收到过任何一帧 /scan 的时刻（哪怕正前方没扫到有效点）—— 硬件自检用它
        # 判断"雷达到底有没有在出数据"。
        self._scan_msg_stamp = 0.0
        # ---- 摔倒靠近：B-2「快照 + move_base 避障 + 短距补齐」----------------
        # 取自参考实现（server_robot3）的策略：快照一次主人的 3D 位置 → 把
        # 「主人前方 standoff 处」换算到 map 系交给 move_base（走 costmap 绕障）
        # → 到位后再用里程计贴最后一段。
        # 关键数字为什么是 0.75 + 0.25：我们 costmap 是 robot_radius 0.25 +
        # inflation_radius 0.5，**move_base 无法把目标规划到离障碍 0.75 m 以内**
        # （躺在地上的人会被低装的雷达当成障碍）。所以导航段停 0.75 m，
        # 最后 0.25 m 用里程计（带 /scan 护栏）贴到 0.5 m。
        self.fall_nav_standoff_distance = max(
            0.45, float(rospy.get_param("~fall_nav_standoff_distance", 0.75))
        )
        self.fall_nav_distance_tolerance = max(
            0.05, float(rospy.get_param("~fall_nav_distance_tolerance", 0.20))
        )
        self.fall_nav_timeout = max(
            5.0, float(rospy.get_param("~fall_nav_timeout", 25.0))
        )
        self.fall_nav_max_travel = max(
            0.0, float(rospy.get_param("~fall_nav_max_travel", 1.50))
        )
        self.fall_nav_frame = str(rospy.get_param("~fall_nav_frame", "map"))
        self.fall_nav_base_frame = str(
            rospy.get_param("~fall_nav_base_frame", "base_footprint")
        )
        # 导航途中的前向硬护栏：只防"真的撞上东西"，所以比目标距离小得多，
        # 否则躺在地上的人（雷达 0.7~1 m 就能看到）会把导航提前掐掉。
        self.fall_nav_guard_distance = max(
            0.20, float(rospy.get_param("~fall_nav_guard_distance", 0.40))
        )
        # ---- 导航段的"看护循环"（2026-10-16 加，移植自 server_robot3 的
        #      send_approach_navigation_goal）--------------------------------
        # 原来的实现只盯雷达护栏 + move_base 最终状态，踩过两个坑：
        #   ① move_base 报"成功"时其实还差一截（定位/局部规划器的误差）；
        #   ② 规划器卡住不动，只能干等到 25 秒超时。
        # 现在导航期间同时看：里程计走了多远（估算剩余）、实际速度（卡没卡住、
        # 是不是已经慢停下来）、以及"有没有进展"，任何一条命中就提前结束这段。
        self.fall_nav_watchdog_enabled = bool(
            rospy.get_param("~fall_nav_watchdog_enabled", True)
        )
        # 慢停接收：剩余距离已经很小，而且"指令慢 + 实际也不动"连续这么多拍，
        # 就当作到位（免得局部规划器在原地反复微调，把 25 秒耗光）。
        self.fall_nav_slow_finish_enabled = bool(
            rospy.get_param("~fall_nav_slow_finish_enabled", True)
        )
        self.fall_nav_slow_finish_speed = max(
            0.005, float(rospy.get_param("~fall_nav_slow_finish_speed", 0.03))
        )
        self.fall_nav_slow_finish_cycles = max(
            1, int(rospy.get_param("~fall_nav_slow_finish_cycles", 3))
        )
        self.fall_nav_slow_finish_tolerance = max(
            0.05,
            float(
                rospy.get_param(
                    "~fall_nav_slow_finish_tolerance",
                    self.fall_nav_distance_tolerance,
                )
            ),
        )
        # move_base 报成功时的校验容差：里程计估算的剩余比它还大就认为"没到位"。
        # 故意放宽到 0.5 m —— 里程计估的是弧长，比直线距离略大，不能卡太死。
        self.fall_nav_arrival_tolerance = max(
            0.10, float(rospy.get_param("~fall_nav_arrival_tolerance", 0.50))
        )
        # 卡死检测：这么久没有进展、而且线速度和角速度都约等于 0 → 判定卡住。
        # 为什么还要看角速度：原地转向是正常的（先转正再走），不算卡死。
        self.fall_nav_stuck_timeout = max(
            2.0, float(rospy.get_param("~fall_nav_stuck_timeout", 6.0))
        )
        self.fall_nav_stuck_min_progress = max(
            0.01, float(rospy.get_param("~fall_nav_stuck_min_progress", 0.05))
        )
        self.fall_nav_stuck_speed = max(
            0.005, float(rospy.get_param("~fall_nav_stuck_speed", 0.025))
        )
        self.fall_nav_stuck_angular_speed = max(
            0.01, float(rospy.get_param("~fall_nav_stuck_angular_speed", 0.05))
        )
        self.fall_nav_odom_speed_max_age = max(
            0.1, float(rospy.get_param("~fall_nav_odom_speed_max_age", 0.6))
        )
        # 发目标前用 /move_base/make_plan 预检：路径太绕（绕路比超限）就放弃，
        # 免得机器人绕过半个房间去"摸"人。
        self.fall_nav_plan_check = bool(rospy.get_param("~fall_nav_plan_check", True))
        # 预检没过时是否**直接放弃**目标：
        #   默认 False —— 现场要求"有障碍就交给 move_base 自己绕"，预检只是个提示，
        #   不该成为拦路虎（实测 make_plan 对某些点位返回空路径，但 move_base 自己能开过去）。
        #   设 True 才恢复"预检不过就换一侧/放弃"的老行为。
        self.fall_nav_plan_block = bool(rospy.get_param("~fall_nav_plan_block", False))
        self.fall_nav_plan_service = str(
            rospy.get_param("~fall_nav_plan_service", "/move_base/make_plan")
        )
        self.fall_nav_plan_detour_ratio = max(
            1.0, float(rospy.get_param("~fall_nav_plan_detour_ratio", 2.0))
        )
        # 最后一段：里程计贴到 0.5 m（导航段停 0.75 → 补 0.25）
        self.fall_extra_close_distance = max(
            0.0, float(rospy.get_param("~fall_extra_close_distance", 0.25))
        )
        self.fall_extra_close_speed = max(
            0.03, min(0.15, float(rospy.get_param("~fall_extra_close_speed", 0.08)))
        )
        self.fall_extra_close_timeout = max(
            1.0, float(rospy.get_param("~fall_extra_close_timeout", 6.0))
        )
        # 补齐段的"盲走上限"：即使复测距离是假的（现场出现过复测 2.73 m 而实际已经很近），
        # 也只允许补这么多。默认 0.25 + 0.15 = 0.40 m。
        self.fall_extra_close_max_distance = max(
            0.05, float(rospy.get_param("~fall_extra_close_max_distance", 0.40))
        )
        # 复测距离可疑的判据：> standoff + 这个值就认为"没量到主人"（跟错人/量到远处），
        # 这时只用默认的 extra_close（0.25 m），不按那个假距离去补。
        self.fall_extra_close_reject_margin = max(
            0.10, float(rospy.get_param("~fall_extra_close_reject_margin", 0.50))
        )
        # 补齐段用**雷达**判断"到人还有多远"：躺着的人在很近时相机会丢，但雷达一直在
        # （0.1 m 高，看得到地上的人）。走到雷达正前方 ≤ 这个距离就停 —— 这是最可靠的
        # "贴近了没有"判据，也方便你调"再靠近一点"。
        #   注意它比硬护栏 fall_scan_guard_distance(0.40) 大，是"软停车线"。
        self.fall_extra_close_scan_stop_distance = max(
            0.25, float(rospy.get_param("~fall_extra_close_scan_stop_distance", 0.55))
        )
        # 相机相对 base_footprint 的前向偏移（URDF 里 kinect2_dock 是 0.145 m）：
        # 点云量的是"到相机"的距离，base 原点在相机后面一点，要补上。
        self.approach_camera_forward_offset = max(
            0.0, float(rospy.get_param("~approach_camera_forward_offset", 0.145))
        )
        # 摔倒路径的取样带：整条身体（15%~85% 行），并且剔掉贴地点的干扰。
        # 为什么不用"上半部分"（站稳/挥手那条用的 10%~60%）：躺着的人身体是长条，
        # 只取上半框等于只取了身体的**某一端**（实测就是"朝头/脚走"）。
        self.fall_owner_band_low = max(
            0.0, min(0.60, float(rospy.get_param("~fall_owner_band_low", 0.15)))
        )
        self.fall_owner_band_high = max(
            self.fall_owner_band_low + 0.10,
            min(1.0, float(rospy.get_param("~fall_owner_band_high", 0.85))),
        )
        self.fall_floor_filter_enabled = bool(
            rospy.get_param("~fall_floor_filter_enabled", True)
        )
        self.fall_floor_clearance = max(
            0.02, float(rospy.get_param("~fall_floor_clearance", 0.12))
        )
        # ---- 摔倒路径的采样方式：躯干锚点（2026-10-11 现场 RViz 实测后加）----
        # 问题：整条身体带采样时，姿态框里"大部分点在家具/墙面上"（用户 RViz 实测），
        #       中位数被拉到远端髋骨与地面交界处 → 机器人朝头/脚走。
        # 做法：用姿态关键点（左右肩+左右髋中点）当种子，只在这个小窗口里取点，
        #       再按"深度一致性"剔掉窗口里落在人**后面**的家具/墙。
        #       锚点拿不到 / 点太少 → 自动退回原来的整条身体带（行为不变）。
        self.fall_torso_anchor_enabled = bool(
            rospy.get_param("~fall_torso_anchor_enabled", True)
        )
        self.fall_torso_keypoint_min_conf = max(
            0.0, min(1.0, float(rospy.get_param("~fall_torso_keypoint_min_conf", 0.35)))
        )
        self.fall_anchor_radius_px = max(
            6.0, float(rospy.get_param("~fall_anchor_radius_px", 22.0))
        )
        self.fall_anchor_depth_band = max(
            0.15, float(rospy.get_param("~fall_anchor_depth_band", 0.70))
        )
        self.fall_anchor_min_points = max(
            5, int(rospy.get_param("~fall_anchor_min_points", 15))
        )
        self._fall_last_sample_source = "band"
        # ---- 摔倒靠近：从"身体侧面"接近（不让机器人冲头/脚）------------------
        # 现场问题：人体斜躺时，沿"机器人→身体中心"那条线开过去，落点会挨着头或脚。
        # 做法：用整条身体带的点估"头↔脚"主方向，取它的**垂直方向**（且朝机器人这一侧）
        # 作为接近方向 → 目标点落在身体侧面，朝向仍正对身体中心；再校验
        # "目标点离任何身体点 ≥ fall_nav_body_clearance"，不够就把 standoff 往外推。
        self.fall_body_axis_enabled = bool(
            rospy.get_param("~fall_body_axis_enabled", True)
        )
        # 默认**关**：侧向接近虽然更"安全"，但它要求机器人一边走一边转（实测目标
        # 朝向比当前朝向差 50°），现场看起来就是"往一边跑"。改成：先原地转正对准
        # 躯干 → 再沿视线靠近，用"离最近身体点保持 standoff"来防止冲头/脚。
        self.fall_side_approach_enabled = bool(
            rospy.get_param("~fall_side_approach_enabled", False)
        )
        # 先转正：bearing 超过这个角度就先原地转，再靠近（用户要求"先正对再靠近"）
        self.fall_face_first_enabled = bool(
            rospy.get_param("~fall_face_first_enabled", True)
        )
        self.fall_face_first_max_angle = max(
            0.05, float(rospy.get_param("~fall_face_first_max_angle", 0.15))
        )
        # 前向护栏触发时：如果主人就在跟前（点云量不到=太近/出画面，或 ≤ 1.6 m），
        # 就当"已经到位"，别再退回 B-1/相机闭环（实测那一退整套动作就乱了）。
        self.fall_nav_guard_is_arrival = bool(
            rospy.get_param("~fall_nav_guard_is_arrival", True)
        )
        self.fall_nav_guard_arrival_distance = max(
            0.5, float(rospy.get_param("~fall_nav_guard_arrival_distance", 1.6))
        )
        # 护栏那个"障碍"要和主人**对得上位置**才认（现场踩过：主人头旁边 20 cm 处
        # 有个纸箱，雷达报的 0.4 m 是箱子不是人）。
        self.fall_nav_guard_match_tolerance = max(
            0.2, float(rospy.get_param("~fall_nav_guard_match_tolerance", 0.6))
        )
        self.fall_body_axis_min_spread = max(
            0.10, float(rospy.get_param("~fall_body_axis_min_spread", 0.20))
        )
        self.fall_body_axis_max_radius = max(
            0.40, float(rospy.get_param("~fall_body_axis_max_radius", 1.10))
        )
        self.fall_nav_body_clearance = max(
            0.20, float(rospy.get_param("~fall_nav_body_clearance", 0.35))
        )
        self.fall_nav_side_extend_max = max(
            0.0, float(rospy.get_param("~fall_nav_side_extend_max", 0.45))
        )
        # ---- 2026-10-12 与用户商定的靠近流程 ---------------------------------
        #   ① 摔倒确认瞬间冻结主人位置（快照）；
        #   ② 先用 move_base 开到"正对主人中心、约 1 m"的中间点（距离不必精确，
        #      有障碍就绕/换一侧）；
        #   ③ 从中间点**直行闭环**（点云测距）走到 ~0.75 m；
        #   ④ 盲走 0.25 m 到 ~0.5 m（带 /scan 软停车线）。
        #   所有阶段 /scan 前向护栏都开着，防止撞到主人或家具。
        self.fall_nav_stage_standoff_distance = max(
            0.60, float(rospy.get_param("~fall_nav_stage_standoff_distance", 1.00))
        )
        self.fall_direct_stop_distance = max(
            0.45, float(rospy.get_param("~fall_direct_stop_distance", 0.75))
        )
        # 中间点被真障碍挡住（现场：主人头边 20 cm 的纸箱）→ 目标点沿"接近方向的
        # 垂直方向"平移这么多，换一侧重试；最多重试几次。
        self.fall_nav_obstacle_retries = max(
            0, int(rospy.get_param("~fall_nav_obstacle_retries", 2))
        )
        self.fall_nav_obstacle_shift = max(
            0.10, float(rospy.get_param("~fall_nav_obstacle_shift", 0.40))
        )
        # 摔倒瞬间冻结的位置：B-2 走不通时，兜底也用它（不再依赖相机重新定位）
        self._fall_frozen_snapshot = None
        # 实地要求（2026-10-12）：B-2 / B-1 都没走通时，**不再**退到"看着人闭环"
        # （那会在原地兜圈找画面），而是原地停下、直接做伸臂动作。
        # 要恢复旧行为就 ~fall_camera_fallback_enabled:=true。
        self.fall_camera_fallback_enabled = bool(
            rospy.get_param("~fall_camera_fallback_enabled", False)
        )
        # 两人场景防串：采样框必须与"主人框"仍有重叠，且面积不能差太多
        self.fall_tracking_min_iou = max(
            0.0, min(0.9, float(rospy.get_param("~fall_tracking_min_iou", 0.05)))
        )
        self.fall_tracking_max_area_ratio = max(
            1.0, float(rospy.get_param("~fall_tracking_max_area_ratio", 1.8))
        )
        self._tf_buffer = None
        self._tf_listener = None
        self._make_plan = None
        # 摔倒靠近的调试可视化：把"主人估计点 / standoff 目标点 / 采样点云"画到 RViz。
        # 只发 Marker，不参与任何决策；默认开，想关就 ~fall_debug_markers:=false。
        self.fall_debug_markers = bool(
            rospy.get_param("~fall_debug_markers", True)
        )
        self.fall_debug_marker_topic = str(
            rospy.get_param("~fall_debug_marker_topic", "/fall_approach/markers")
        )
        self._fall_debug_pub = None
        self._fall_debug_warned = False
        self._fall_debug_logged = False
        self.fall_assist_arm_enabled = bool(
            rospy.get_param("~fall_assist_arm_enabled", True)
        )
        self.mani_ctrl_topic = rospy.get_param(
            "~mani_ctrl_topic", "/wpb_home/mani_ctrl"
        )
        # 这些数值取自旧包（同一台机器人上实测用过的那一套）
        self.fall_assist_arm_extend_lift = float(
            rospy.get_param("~fall_assist_arm_extend_lift", 0.50)
        )
        self.fall_assist_arm_extend_gripper = float(
            rospy.get_param("~fall_assist_arm_extend_gripper", 0.12)
        )
        self.fall_assist_arm_retract_lift = float(
            rospy.get_param("~fall_assist_arm_retract_lift", 0.0)
        )
        self.fall_assist_arm_retract_gripper = float(
            rospy.get_param("~fall_assist_arm_retract_gripper", 0.12)
        )
        self.fall_assist_arm_lift_velocity = float(
            rospy.get_param("~fall_assist_arm_lift_velocity", 0.5)
        )
        self.fall_assist_arm_gripper_velocity = float(
            rospy.get_param("~fall_assist_arm_gripper_velocity", 5.0)
        )
        self.fall_assist_arm_extend_wait = max(
            0.0, float(rospy.get_param("~fall_assist_arm_extend_wait", 3.0))
        )
        self.fall_assist_arm_hold_seconds = max(
            0.0, float(rospy.get_param("~fall_assist_arm_hold_seconds", 4.0))
        )
        self.fall_assist_arm_retract_wait = max(
            0.0, float(rospy.get_param("~fall_assist_arm_retract_wait", 3.0))
        )
        self.fall_assist_arm_completion_wait = max(
            0.0, float(rospy.get_param("~fall_assist_arm_completion_wait", 1.0))
        )
        self.fall_assist_arm_command_rate = max(
            0.5, float(rospy.get_param("~fall_assist_arm_command_rate", 5.0))
        )
        self.fall_assist_done_text = rospy.get_param(
            "~fall_assist_done_text", "我已经把机械臂伸过去了，请慢一点起来。"
        )
        self.mani_ctrl_pub = None

        # 动作结论由本文件播报（要带姓名），所以关掉动作节点自己那句
        self.action_speak_enabled = bool(
            rospy.get_param("~action_speak_enabled", False)
        )

        self.search_start_text = rospy.get_param("~search_start_text", "开始寻找主人。")
        # 注册完成 → "请志愿者就位" + 原地停 N 秒 → 再开始逐房间寻找（2026-10-15 现场要求）。
        # 两个参数都可调：~search_ready_text 设成空串就不播；~search_ready_pause 设 0 就不停。
        self.search_ready_text = rospy.get_param(
            "~search_ready_text", "请志愿者就位，我即将开始寻找。"
        )
        self.search_ready_pause = max(
            0.0, float(rospy.get_param("~search_ready_pause", 3.0))
        )
        # 开局先去哪个航点注册（2026-10-15 新流程）：例如 living_room。
        # 空串 = 老流程（在起点原地注册）。到不了注册点也会继续就地注册，不中断。
        self.start_waypoint = str(
            rospy.get_param("~start_waypoint", "") or ""
        ).strip()
        self.start_waypoint_text = rospy.get_param(
            "~start_waypoint_text", "正在前往%s进行注册。"
        )
        self.search_each_text = rospy.get_param("~search_each_text", "正在寻找%s。")
        self.owner_action_start_text = rospy.get_param(
            "~owner_action_start_text", "找到主人%s了，现在开始检测您的动作。"
        )
        # 动作识别要 5~10 秒。中间给一句提示，否则对比 ③e（动作节点自己会说
        # "正在识别，请稍候"）会觉得 ③g 卡住了。wait=False：这句话是异步播放的，
        # 正好和"拉起/预热动作节点"并行，几乎不占额外时间。设成空串可关掉。
        self.owner_action_wait_text = rospy.get_param(
            "~owner_action_wait_text", "正在识别，请稍候。"
        )
        self.owner_missing_text = rospy.get_param("~owner_missing_text", "没有找到%s。")
        self.all_done_text = rospy.get_param("~all_done_text", "全部主人检查完毕。")
        # 找人顺序：
        #   true （默认）= 谁先出现就先认谁；认过的这一轮不再找；转一圈没认到别人
        #                 就播报剩下的人"没找到"
        #   false       = 按注册顺序一位一位找（③f 最早的行为）
        self.search_any_order = bool(
            rospy.get_param("~search_any_order", True)
        )
        # 动作结论拿到之后的固定等待。父类默认 1.0 秒，是留给"动作节点自己
        # 把那句话说完整"的；但本任务把动作节点的播报关了（speak_enabled=false），
        # 结论由本文件自己播，所以这 1 秒是纯浪费 —— 压到 0.2 秒。
        self.action_speech_grace = max(
            0.0, float(rospy.get_param("~action_speech_grace", 0.2))
        )

        # ---- 相机看门狗 ----
        # 实测踩过一次坑：Kinect 的画面流会中途自己断掉（驱动日志里
        # publishing rate 变成 0Hz），而进程、TCP 连接都还在。上层只看到
        # "没有新画面"，就报成"没检测到人 / 没有检测到人脸"，很容易被当成
        # 识别算法的问题，白白浪费现场时间。这里单独盯住画面新鲜度，
        # 一旦超时就明确报警，并在报错时给出准确原因。
        self.camera_watchdog_enabled = bool(
            rospy.get_param("~camera_watchdog_enabled", True)
        )
        self.camera_stale_seconds = max(
            1.0, float(rospy.get_param("~camera_stale_seconds", 3.0))
        )
        # 连续断流多久才认定为"掉线"（避免画面偶尔卡一两帧就报警）
        self.camera_lost_grace_seconds = max(
            0.0, float(rospy.get_param("~camera_lost_grace_seconds", 2.0))
        )
        self.camera_lost = False
        self._camera_watchdog_stop = False
        # 掉线后的处理策略：看门狗的恢复梯子（重载 nodelet → 重启 nodelet
        # manager → USB 软拔插）通常 20~60 秒能把画面拉回来。这段时间里如果
        # 继续转圈/开车，识别全是空的，8 分钟预算会被白白烧掉，日志还会被
        # 误读成"没找到人"。所以默认**原地等**，等不回来再明确报错。
        self.camera_hold_on_loss = bool(rospy.get_param("~camera_hold_on_loss", True))
        self.camera_recovery_wait_seconds = max(
            5.0, float(rospy.get_param("~camera_recovery_wait_seconds", 90.0))
        )
        # 起栈后到"第一次出图"允许等多久（看门狗可能正在做恢复，别 35 秒就判死）。
        self.camera_startup_timeout = max(
            5.0, float(rospy.get_param("~camera_startup_timeout", 90.0))
        )

        # 父类认到主人会单独播报"识别到主人X"；这里抑制掉，合并进上面那句
        self._suppress_owner_found_speech = False
        self._feedback_module = None

    # ------------------------------------------------------------------
    # 相机看门狗
    # ------------------------------------------------------------------
    def image_stale_seconds(self):
        """最近一帧画面距今多少秒；从没收到过画面返回 None。"""
        stamp = getattr(self, "latest_image_stamp", None)
        if stamp is None:
            return None
        return time.time() - stamp

    def start_camera_watchdog(self):
        """后台盯着相机画面是否还在更新（断流时明确报警一次）。"""
        if not self.camera_watchdog_enabled:
            rospy.loginfo("相机看门狗已关闭（camera_watchdog_enabled=false）")
            return

        def worker():
            stale_since = None
            reported = False
            while (
                not self._camera_watchdog_stop
                and not rospy.is_shutdown()
            ):
                age = self.image_stale_seconds()
                now = time.time()
                if age is None or age > self.camera_stale_seconds:
                    if stale_since is None:
                        stale_since = now
                    elif not reported and now - stale_since >= self.camera_lost_grace_seconds:
                        reported = True
                        self.camera_lost = True
                        # 报真实的断流时长（age = 距最近一帧的秒数）。以前这里打的是
                        # "从判定开始起算"的宽限时间，看着只有 2 秒，容易低估故障。
                        outage = age if age is not None else (now - stale_since)
                        rospy.logerr(
                            "相机掉线：%s 已连续 %.1f 秒没有新画面"
                            "（进程还活着，是画面流断了，不是识别算法的问题）。"
                            "看门狗会自己救：重载 bridge nodelet → 重启整个 nodelet "
                            "manager → USB 软拔插；画面回来之前先别急着判任务失败",
                            self.image_topic,
                            outage,
                        )
                        try:
                            self.speak(
                                "摄像头没有画面了，请检查摄像头的连接。",
                                wait=False,
                            )
                        except Exception:
                            pass
                else:
                    if reported:
                        rospy.loginfo("相机画面已恢复（%.2f 秒前有新帧）", age)
                    stale_since = None
                    reported = False
                time.sleep(0.5)

        thread = threading.Thread(target=worker, name="camera-watchdog")
        thread.daemon = True
        thread.start()
        rospy.loginfo(
            "相机看门狗已启动：超过 %.1f 秒没有新画面就报警", self.camera_stale_seconds
        )

    def stop_camera_watchdog(self):
        self._camera_watchdog_stop = True

    def hold_until_camera_recovers(self, phase, timeout=None):
        """相机断流时：原地停下，等看门狗把画面救回来。

        返回 True=画面恢复/本来就有画面；False=超时（调用方按失败处理）。

        父类扫描循环会用 getattr 调这个方法；只有本任务类有定义，所以旧包 /
        调试脚本的行为完全不变。
        """
        if not self.camera_hold_on_loss:
            return True
        age = self.image_stale_seconds()
        # 从没收到过帧时（age is None）不在这里等：那是"相机还没起来"，由
        # wait_for_camera_inputs 负责。
        if age is None or age <= self.camera_stale_seconds:
            return True

        limit = float(
            self.camera_recovery_wait_seconds if timeout is None else timeout
        )
        started = time.time()
        try:
            self.stop_base()
        except Exception:
            pass
        rospy.logwarn(
            "相机断流（已 %.1f 秒没有新帧）：暂停%s，等看门狗恢复画面（最多 %.0f 秒）",
            age,
            phase,
            limit,
        )
        while not rospy.is_shutdown() and time.time() - started < limit:
            age = self.image_stale_seconds()
            if age is not None and age <= self.camera_stale_seconds:
                rospy.loginfo(
                    "相机画面已恢复（等了 %.1f 秒），继续%s", time.time() - started, phase
                )
                return True
            rospy.sleep(0.5)
        rospy.logerr(
            "等了 %.0f 秒相机仍未恢复，无法继续%s。"
            "现场处理：先跑 tools/camera_health.sh 看结论，再按提示做 USB 软拔插 / "
            "人工重插 12V",
            limit,
            phase,
        )
        return False

    # ------------------------------------------------------------------
    # 播报：合并"找到主人"那一句
    # ------------------------------------------------------------------
    def speak(self, text, wait=True):
        if getattr(self, "_suppress_owner_found_speech", False):
            rospy.loginfo("TTS（已抑制，稍后合并播报）: %s", text)
            return
        return super().speak(text, wait=wait)

    def announce_owner_result(self, result):
        """父类在这里播报"识别到主人X"：抑制声音，其余动作照旧。"""
        self._suppress_owner_found_speech = True
        try:
            return super().announce_owner_result(result)
        finally:
            self._suppress_owner_found_speech = False

    # ------------------------------------------------------------------
    # 播报文案：复用 feedback_after_recognition.py 的纯逻辑
    # ------------------------------------------------------------------
    def feedback_module(self):
        if self._feedback_module is None:
            path = os.path.join(_SCRIPT_DIR, "feedback_after_recognition.py")
            spec = importlib.util.spec_from_file_location(
                "wpb_feedback_after_recognition", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._feedback_module = module
        return self._feedback_module

    def action_speech_for(self, action, place, owner_name):
        """动作 + 地点 + 姓名 → 一句播报，例如"主人张三正在躺着。"。"""
        try:
            feedback = self.feedback_module().feedback_for_action(
                action, place, owner_name
            )
            return feedback.speech
        except Exception as exc:
            rospy.logwarn("拼动作播报失败，退回简单说法：%s", exc)
            return "主人%s的动作是%s。" % (owner_name or "", action or "unknown")

    def owner_feedback(self, action, place, owner_name):
        """动作 → (播报文案, 下一步动作名)。

        下一步动作名由 feedback_after_recognition.py 统一定义，
        是"主流程"和"后续功能模块"之间的约定：
          sitting / lying  -> next_waypoint（**2026-10-16 现场改的**：
                              不追问电器开关，播报完就直接继续下一个航点）
          fallen / sudden_fall -> approach_and_assist
          waving           -> approach_and_ask_help
          unknown          -> retry

        播报里会带上**当前所在房间**（由当前航点名映射成中文，
        例如 living_room → 客厅）："主人张三正在客厅坐着"。
        """
        try:
            feedback = self.feedback_module().feedback_for_action(
                action,
                place,
                owner_name,
                room=self.current_room_speech_name(),
            )
            return feedback.speech, feedback.next_step
        except Exception as exc:
            rospy.logwarn("拼动作播报失败，退回简单说法：%s", exc)
            return (
                "主人%s的动作是%s。" % (owner_name or "", action or "unknown"),
                "unknown",
            )

    # ------------------------------------------------------------------
    # 导航：真正调用 move_base 走到航点
    # ------------------------------------------------------------------
    # 为什么这里要自己包一层，而不直接用父类的 navigate_to_waypoint()：
    #   父类那个是"要么成功、要么抛异常"，适合"找不到路就直接结束"的老流程。
    #   总任务不能这样 —— 一个房间导航失败就整场中断，太亏。这里的规矩是：
    #     · 任何失败都只返回 False，绝不抛异常；
    #     · 连续失败到上限就彻底不导航，退回"就地找"；
    #     · 时间不够就不出发，把时间留给"回出口"。
    @staticmethod
    def parse_name_list(value):
        """把 "a,b,c" 或 ["a","b"] 这种参数统一解析成 ["a","b","c"]。"""
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = value.replace(";", ",").replace("，", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]
        return [str(item).strip() for item in raw_values if str(item).strip()]

    @staticmethod
    def parse_fallback_waypoints(value):
        """解析"备用航点"参数：`bedroom2=bedroom,kitchen2=kitchen` → {备用: 主航点}。

        语义：只有当**主航点那一间一个人都没找到**时，才去备用航点；
        主航点找到了人（哪怕还有别的主人没找到）就跳过备用航点。
        分隔符 = 或 : 都行。
        """
        mapping = {}
        if isinstance(value, (list, tuple)):
            items = [str(item) for item in value]
        else:
            items = str(value or "").replace("；", ";").replace("，", ",").split(",")
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            if "=" in text:
                fallback, anchor = text.split("=", 1)
            elif ":" in text:
                fallback, anchor = text.split(":", 1)
            else:
                continue
            fallback = fallback.strip()
            anchor = anchor.strip()
            if fallback and anchor:
                mapping[fallback] = anchor
        return mapping

    @staticmethod
    def parse_float_map(value):
        """把 "纸团:0.18,空瓶子:0.25" 或 {"纸团":0.18} 解析成 {中文名: 分数}。"""
        if value is None:
            return {}
        if isinstance(value, dict):
            items = list(value.items())
        elif isinstance(value, str):
            items = []
            for chunk in value.replace("；", ";").replace("，", ",").replace(";", ",").split(","):
                if not chunk.strip():
                    continue
                key, _sep, raw = chunk.partition(":")
                if _sep:
                    items.append((key, raw))
        else:
            items = []
        parsed = {}
        for key, raw in items:
            name = str(key).strip()
            if not name:
                continue
            try:
                parsed[name] = float(raw)
            except (TypeError, ValueError):
                continue
        return parsed

    def waypoint_speech_name(self, waypoint_name):
        """航点名 → 播报用的中文名；没配过就原样返回。

        "同一个房间的另一个点位"（bedroom2 / kitchen_2 这种带尾部数字的）会先去掉
        尾部数字再查一次：bedroom2 → bedroom → "卧室"。
        这样 bedroom2 播报的是"正在前往卧室寻找主人"，不会冒出"次卧"。
        如果某个点位想单独叫别的名字，在 waypoint_speech_names 里给它加一条即可
        （显式配置优先于"去数字"）。
        """
        key = str(waypoint_name or "").strip()
        if not key:
            return key
        if key in self.waypoint_speech_names:
            return self.waypoint_speech_names[key]
        stripped = ""
        for index in range(len(key) - 1, -1, -1):
            if key[index].isdigit() or key[index] == "_":
                continue
            stripped = key[: index + 1]
            break
        if stripped and stripped in self.waypoint_speech_names:
            return self.waypoint_speech_names[stripped]
        return key

    def current_room_speech_name(self):
        """当前所在房间的中文名（动作播报用）：主人张三正在**客厅**坐着。

        来源是"当前所在航点"（run_waypoint_search_round 每进一间就更新），
        经 waypoint_speech_names 映射；没进过房间（原地测）就返回空串，
        播报退回不带房间的老格式。
        """
        key = str(getattr(self, "current_room_waypoint", "") or "").strip()
        if not key:
            return ""
        return self.waypoint_speech_name(key)

    # ---- 开局自动设置 AMCL 初始位姿（替代手点 2D Pose Estimate）----
    def publish_auto_initial_pose(self):
        """把"机器人开机所在航点"的位姿发到 /initialpose，让 AMCL 立刻对上。

        为什么可以这么做：比赛流程固定 —— 机器人就停在门口（= 回出口用的
        `exit` 航点）起步。把这个航点在 map 系里的位姿当成初始位姿发出去，
        等价于人在 RViz 里点一次 2D Pose Estimate 并把箭头拖到同一个位置。

        航点表里没有对应航点 / 发不出去时只打 WARN，不中断流程
        （那时照旧手动点一次就行）。
        """
        waypoint_name = (
            self.auto_initial_pose_waypoint or self.exit_waypoint_name or ""
        ).strip()
        pose = None
        if waypoint_name:
            try:
                pose = self.load_waypoint_pose(waypoint_name)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn(
                    "自动初始位姿：读航点 %s 失败（%s），跳过", waypoint_name, exc
                )
        if pose is None:
            rospy.logwarn(
                "自动初始位姿：没有可用的航点（%s）→ 跳过，需要时手动点 2D Pose Estimate",
                waypoint_name or "（未配置）",
            )
            return False
        try:
            from geometry_msgs.msg import PoseWithCovarianceStamped
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn("自动初始位姿：导入消息类型失败（%s），跳过", exc)
            return False
        if self._initial_pose_pub is None:
            self._initial_pose_pub = rospy.Publisher(
                "/initialpose", PoseWithCovarianceStamped, queue_size=1
            )
        message = PoseWithCovarianceStamped()
        message.header.frame_id = self.fall_nav_frame
        message.pose.pose.position.x = float(pose["x"])
        message.pose.pose.position.y = float(pose["y"])
        message.pose.pose.position.z = 0.0
        message.pose.pose.orientation.x = 0.0
        message.pose.pose.orientation.y = 0.0
        message.pose.pose.orientation.z = float(pose["z"])
        message.pose.pose.orientation.w = float(pose["w"])
        # 协方差照抄 RViz 的 2D Pose Estimate：位置 0.25 m²、朝向 0.0685 rad²
        covariance = [0.0] * 36
        covariance[0] = 0.25
        covariance[7] = 0.25
        covariance[35] = 0.0685
        message.pose.covariance = covariance
        # 发 3 次（AMCL 有时候第一帧还没订阅上），然后等一下让它收敛
        for _ in range(3):
            message.header.stamp = rospy.Time.now()
            try:
                self._initial_pose_pub.publish(message)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn("自动初始位姿：发布失败（%s）", exc)
                return False
            rospy.sleep(0.3)
        rospy.loginfo(
            "已自动设置 AMCL 初始位姿 = 航点 %s 的位姿（x=%.2f y=%.2f）；"
            "等 %.1f 秒让它收敛",
            waypoint_name,
            float(pose["x"]),
            float(pose["y"]),
            self.auto_initial_pose_wait,
        )
        if self.auto_initial_pose_wait > 0.0:
            rospy.sleep(self.auto_initial_pose_wait)
        # 读一次 /amcl_pose 做对照，日志里能看出对不对得上
        try:
            from geometry_msgs.msg import PoseWithCovarianceStamped as _Pose

            echoed = rospy.wait_for_message("/amcl_pose", _Pose, timeout=3.0)
            position = echoed.pose.pose.position
            rospy.loginfo(
                "AMCL 现在认为机器人在 x=%.2f y=%.2f（设定值 x=%.2f y=%.2f，"
                "差 %.2f m）",
                position.x,
                position.y,
                float(pose["x"]),
                float(pose["y"]),
                math.hypot(position.x - float(pose["x"]), position.y - float(pose["y"])),
            )
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(5.0, "读 /amcl_pose 对照失败：%s", exc)
        return True

    def report_manual_initial_pose(self):
        """_auto_initial_pose:=false 时的开局提醒 + 位姿自检。

        关掉自动初值以后，"机器人现在在哪"完全靠人在 RViz 里点 2D Pose Estimate。
        点没点、点得对不对，日志里得能看出来，否则一起导航就又是
        "Failed to get a plan." 白排查一轮。所以这里只做两件事：

          1. 把当前 /amcl_pose 打出来，并在离最近航点超过 1.0 m 时提醒"可能点偏了"；
          2. 提醒这一步是手动的，别忘。

        只读，不发任何东西；收不到位姿也只是 WARN，不中断流程。
        """
        names = []
        for name in [self.start_waypoint, self.exit_waypoint_name] + list(
            self.search_waypoints
        ):
            name = str(name or "").strip()
            if name and name not in names:
                names.append(name)
        try:
            from geometry_msgs.msg import PoseWithCovarianceStamped

            echoed = rospy.wait_for_message(
                "/amcl_pose", PoseWithCovarianceStamped, timeout=3.0
            )
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn(
                "自动初始位姿已关闭（_auto_initial_pose:=false），但读 /amcl_pose 失败：%s。"
                "跑任务前请确认已经在 RViz 里点过 2D Pose Estimate。",
                exc,
            )
            return False
        position = echoed.pose.pose.position
        quaternion = echoed.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
        )
        nearest = ""
        nearest_distance = None
        for name in names:
            try:
                pose = self.load_waypoint_pose(name)
            except Exception:  # noqa: BLE001
                continue
            distance = math.hypot(position.x - pose["x"], position.y - pose["y"])
            if nearest_distance is None or distance < nearest_distance:
                nearest, nearest_distance = name, distance
        if nearest:
            rospy.loginfo(
                "自动初始位姿已关闭（_auto_initial_pose:=false），用手点的 2D Pose Estimate："
                "AMCL 现在认为机器人在 x=%.2f y=%.2f yaw=%.2f（离最近航点 %s 差 %.2f m）",
                position.x,
                position.y,
                yaw,
                nearest,
                nearest_distance,
            )
            if nearest_distance > 1.0:
                rospy.logwarn(
                    "当前位姿离最近的航点 %s 还有 %.2f m（>1.0 m）。"
                    "如果车其实停在门口，这个值偏大说明 2D Pose Estimate 点歪了 —— "
                    "导航会照常失败，先重设一次再跑。",
                    nearest,
                    nearest_distance,
                )
        else:
            rospy.logwarn(
                "自动初始位姿已关闭（_auto_initial_pose:=false）：AMCL 现在认为机器人在 "
                "x=%.2f y=%.2f，但没能读到任何航点做对照（检查 %s）",
                position.x,
                position.y,
                self.waypoint_file,
            )
        return True

    # ---- 比赛时间预算 ----
    def competition_time_remaining(self):
        """距比赛时限还剩多少秒；还没开始计时返回 inf。

        2026-10-16 现场要求：默认**关掉"用时长裁剪流程"**——
        这里直接返回 inf，于是下游全部失去约束：
          · competition_should_wrap_up() 恒为假（不会因为"时间不够"跳过房间/提前收尾）；
          · navigation_timeout_for() = 固定单航点超时（不再被剩余时间压缩）；
          · room_scan_budget() = 固定单间扫描时长。
        想恢复旧行为：起栈传 _competition_time_budget_enabled:=true。
        """
        if not getattr(self, "competition_time_budget_enabled", False):
            return float("inf")
        if self.competition_started_time is None:
            return float("inf")
        return self.competition_total_seconds - (
            time.time() - self.competition_started_time
        )

    def competition_should_wrap_up(self):
        """是否该停止接新任务（新房间 / 下一位主人），转入离场收尾。"""
        return self.competition_time_remaining() <= self.competition_exit_reserve_seconds

    def log_time_budget(self, stage):
        if not getattr(self, "competition_time_budget_enabled", False):
            return   # 按现场要求：不再打印"已用/剩余"这些行
        remaining = self.competition_time_remaining()
        if remaining == float("inf"):
            return
        rospy.loginfo(
            "比赛时间预算 | %s | 已用 %.1fs | 剩余 %.1fs（离场预留 %.1fs）",
            stage,
            self.competition_total_seconds - remaining,
            remaining,
            self.competition_exit_reserve_seconds,
        )

    # ---- 导航失败计数 ----
    def note_navigation_failure(self, reason):
        """记一次导航失败；**同一目标**连续失败到上限就放弃这个目标的导航。

        定位坏掉时（AMCL 没收敛、机器人位姿和地图对不上）每次导航都要干等一个
        超时，4 个房间就能把 8 分钟全吃光。到上限后这一间退回"就地找"：虽然没有
        导航分，但认主人 / 动作 / 交互那部分的分还能拿；**下一间重新计数、继续导航**
        （2026-10-16 现场要求：别再出现"前两间失败 → 后面所有房间都原地转"）。
        """
        self.navigation_failure_count += 1
        self.navigation_failure_total += 1
        rospy.logwarn(
            "导航失败 %d/%d（本目标；全场累计 %d）（%s）",
            self.navigation_failure_count,
            self.navigation_failure_limit,
            self.navigation_failure_total,
            reason,
        )
        if self.navigation_gave_up():
            rospy.logerr(
                "导航连续失败 %d 次，**本目标**不再尝试导航，就地识别"
                "（下一间房会重新计数、照常导航）。"
                "如果每间都这样，请检查 AMCL 初始位姿 / 地图是否和实际一致、"
                "通道里是否有人或家具挡住了"
                "（当前 costmap：inflation_radius 0.40 m / robot_radius 0.25 m）",
                self.navigation_failure_count,
            )

    def navigation_gave_up(self):
        """当前目标的连续失败次数到上限（每个新目标开始时会清零）。"""
        return self.navigation_failure_count >= self.navigation_failure_limit

    def reset_room_navigation_failures(self, target_name=""):
        """开始一个新目标（每间房 / 起点 / 回出口）前，把"本目标连续失败"清零。

        2026-10-16：失败计数从"整场"改成"按目标"，避免一间房卡住就让后面所有房间
        都失去导航（现场表现是"后面全在原地乱转"）。
        """
        if self.navigation_failure_count:
            rospy.loginfo(
                "导航失败计数清零（上一目标 %d 次失败，全场累计 %d 次）%s",
                self.navigation_failure_count,
                self.navigation_failure_total,
                ("，新目标：%s" % target_name) if target_name else "",
            )
        self.navigation_failure_count = 0

    # ---- 真正发目标 ----
    def navigation_timeout_for(self):
        """本次导航最多等多少秒：取"单航点超时"和"剩余可用时间"里的小者。"""
        budget = self.waypoint_goal_timeout
        remaining = self.competition_time_remaining()
        if remaining != float("inf"):
            budget = min(
                budget, max(0.0, remaining - self.competition_exit_reserve_seconds)
            )
        return budget

    def wait_for_navigation_goal(self, wait_timeout):
        """轮询等待 move_base 结果（不用 wait_for_result 一把阻塞）。

        好处是等待期间还能刷新 YOLO 调试窗口 —— 否则导航那十几秒画面是黑的，
        看起来像程序卡死了。
        """
        deadline = time.time() + max(0.1, float(wait_timeout))
        while not rospy.is_shutdown():
            remaining = deadline - time.time()
            if remaining <= 0.0:
                return False
            if self.move_base.wait_for_result(rospy.Duration(min(0.2, remaining))):
                return True
            self.update_yolo_window("导航中")
        return False

    def send_navigation_goal(self, goal, timeout=None, label=None):
        """发一个 move_base 目标并等结果；返回 (是否成功, 失败原因)。不抛异常。"""
        if timeout is None:
            wait_timeout = self.navigation_timeout_for()
        else:
            wait_timeout = max(0.1, float(timeout))
        target_label = label or self.waypoint_name
        self.move_base.send_goal(goal)
        if not self.wait_for_navigation_goal(wait_timeout):
            self.move_base.cancel_goal()
            self.stop_base()
            self.note_navigation_failure("超时")
            return False, "导航到 %s 超时（等了 %.0f 秒）" % (
                target_label,
                wait_timeout,
            )
        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            self.stop_base()
            self.note_navigation_failure("move_base 状态 %s" % state)
            return False, "导航到 %s 失败，move_base 状态 %s" % (
                target_label,
                state,
            )
        self.navigation_failure_count = 0
        return True, ""

    def navigate_to_waypoint(self, waypoint_name=""):
        """导航到指定航点。返回 True=已经在（或到）那里；False=没到。

        永远不会抛异常：调用方拿到 False 就"就地干原来的事"，流程不中断。
        """
        target_name = str(waypoint_name or "").strip()
        if not self.navigate_enabled:
            rospy.loginfo("导航未启用（navigate_enabled=false），就地不动")
            return True
        if not target_name:
            rospy.logwarn("没有指定航点，就地不动")
            return False
        if self.navigate_dry_run:
            # 空跑：一步都不开，直接报"到了"。注意这不只是省时间 ——
            # 地图/航点和现场对不上时，真开过去反而可能撞东西或把自己开丢，
            # 空跑能让你安全地把"到达之后"的整套流程测完。
            rospy.logwarn(
                "空跑模式：不移动，直接当作已到达 %s（%s）",
                target_name,
                self.waypoint_speech_name(target_name),
            )
            return True
        if self.navigation_gave_up():
            rospy.logwarn(
                "本目标导航已因连续失败跳过（%d 次），就地寻找"
                "（下一个目标会重新计数）",
                self.navigation_failure_count,
            )
            return False

        # 时间不够就别出发：走一半被时间预算掐断，比不去还亏。
        budget = self.navigation_timeout_for()
        if budget < self.min_navigation_seconds:
            rospy.logwarn(
                "剩余时间只够 %.0f 秒导航（< %.0f 秒），放弃前往 %s，就地寻找",
                budget,
                self.min_navigation_seconds,
                target_name,
            )
            return False
        # 这一趟（含失败重试）的硬性截止时刻
        deadline = time.time() + budget

        if not self.move_base.wait_for_server(
            rospy.Duration(self.move_base_server_timeout)
        ):
            self.note_navigation_failure("move_base 动作服务不可用")
            return False

        try:
            pose = self.load_waypoint_pose(target_name)
        except Exception as exc:
            self.note_navigation_failure("航点读取失败：%s" % exc)
            return False

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = pose["x"]
        goal.target_pose.pose.position.y = pose["y"]
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.z = pose["z"]
        goal.target_pose.pose.orientation.w = pose["w"]

        self.clear_move_base_costmaps("去 %s 之前" % target_name)
        rospy.loginfo(
            "前往航点 %s（%s）：x=%.2f y=%.2f，最多等 %.0f 秒",
            target_name,
            self.waypoint_speech_name(target_name),
            pose["x"],
            pose["y"],
            budget,
        )
        success, error_message = self.send_navigation_goal(
            goal, timeout=budget, label=target_name
        )
        if (
            not success
            and self.retry_navigation_after_clear
            and not self.navigation_gave_up()
        ):
            # 老规矩：失败多半是局部代价地图里堆了障碍（人走过、激光噪点），
            # 清掉重发同一个目标往往就好了。
            # ⚠️ 重试只能在**同一个预算**里花剩下的时间，不能各自再给 45 秒 ——
            # 否则一个房间最坏能吃掉 90 秒，4 个房间就把 8 分钟烧穿。
            leftover = deadline - time.time()
            if leftover >= self.min_navigation_seconds:
                rospy.logwarn(
                    "%s；清代价地图后，用剩下的 %.0f 秒重试一次",
                    error_message,
                    leftover,
                )
                self.clear_move_base_costmaps("导航失败后")
                goal.target_pose.header.stamp = rospy.Time.now()
                success, error_message = self.send_navigation_goal(
                    goal, timeout=leftover, label=target_name
                )
            else:
                rospy.logwarn("%s；剩余时间不足，不再重试", error_message)

        if not success:
            rospy.logwarn("没到 %s：%s", target_name, error_message)
            return False
        self.stop_base()
        rospy.loginfo(
            "到达航点 %s（%s）", target_name, self.waypoint_speech_name(target_name)
        )
        return True

    def navigate_to_search_position(self, waypoint_name=""):
        """找主人之前"走到位"的那一步。返回 True=可以就地开始扫描。"""
        target_name = str(waypoint_name or "").strip()
        if not target_name:
            rospy.loginfo("不留航点（原地开始寻找主人）")
            return True
        return self.navigate_to_waypoint(target_name)

    # ------------------------------------------------------------------
    # 地面垃圾识别（"找到垃圾"60 分）—— 只找 + 播报，不抓取
    # ------------------------------------------------------------------
    def set_trash_detector_paused(self, paused, reason=""):
        """暂停/恢复垃圾桶检测节点的推理（不关进程：模型常驻，切换没有启动开销）。

        为什么要按阶段开关：yolov8s-world 在 CPU 上一次推理要 98ms 墙钟、
        但吃掉约 780ms 的 CPU 时间（PyTorch 默认开 8 个线程，等于瞬间占满 8 个核）。
        每 3 帧推理一次（相机 15Hz -> 5 次/秒）就是**持续约 4 个核**的开销 ——
        而主任务的姿态/ReID/人脸也在同一个 CPU 上跑。所以只在真要看垃圾的阶段开。
        """
        if not self.trash_node_pause_topic or not self.trash_announce_enabled:
            return False
        paused = bool(paused)
        if self.trash_detector_paused == paused:
            return True
        if self.trash_pause_pub is None:
            try:
                self.trash_pause_pub = rospy.Publisher(
                    self.trash_node_pause_topic, Bool, queue_size=1
                )
                rospy.sleep(0.3)  # 等订阅端连上，否则第一条会丢
            except Exception as exc:
                rospy.logwarn("创建垃圾检测 pause 发布器失败：%s", exc)
                self.trash_node_pause_topic = ""
                return False
        self.trash_pause_pub.publish(Bool(data=paused))
        self.trash_detector_paused = paused
        rospy.loginfo(
            "垃圾检测节点：%s%s",
            "暂停" if paused else "恢复",
            "（%s）" % reason if reason else "",
        )
        return True

    def detections_callback(self, message):
        """收到一帧检测框：先交给父类（主人识别要用），再顺手记一笔垃圾候选。

        在这里累积是**零额外耗时**的：找主人 / 居中 / 认动作时相机一直在扫这个
        房间，检测框本来就会流过来。父类先跑完再记，避免嵌套拿同一把锁。
        （如果垃圾单独配了话题，这一路就只管主人识别，累积在另一个回调里做。）
        """
        super().detections_callback(message)
        if not self.trash_announce_enabled:
            return
        if (
            self.trash_detections_topic
            and self.trash_detections_topic != self.detections_topic
        ):
            return
        try:
            self.note_trash_detections(time.time())
        except Exception as exc:  # noqa: BLE001 - 垃圾只是加分项，绝不能拖垮主流程
            rospy.logwarn_throttle(5.0, "累积垃圾候选出错（已忽略）：%s", exc)

    def trash_detections_callback(self, message):
        """垃圾检测单独走一路时，收到一帧就记一笔。"""
        self.trash_detections = list(message.detections)
        self.trash_detections_stamp = time.time()
        if not self.trash_announce_enabled:
            return
        try:
            self.note_trash_detections(self.trash_detections_stamp)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(5.0, "累积垃圾候选出错（已忽略）：%s", exc)

    def trash_min_score_for(self, chinese_name):
        """某类垃圾的分数阈值；没单独配就用全局值。"""
        return float(
            self.trash_min_score_by_kind.get(chinese_name, self.trash_min_score)
        )

    def classify_trash_detection(self, detection, image_height, min_score):
        """垃圾候选判据（顺序与 cpy 一致）：返回 (是否拒绝的原因, 指标)。

        判据顺序 分数 → 框大小 → 是否在画面下半部分（"在地上"）：
          · 尺寸看**长边**而不是短边。瓶子是细长的：6cm 宽 / 25cm 高的瓶子在
            3m 处框只有 11x44px，按短边判就永远"框太小"。现场 3053 次瓶子
            检测里有 2123 次（70%）就是这么被丢掉的。
          · 另外单独拦 1~2px 宽的细条（噪声，不是物体）。
          · center_y_ratio：垃圾必须出现在画面下半部分。相机低头装着，
            桌上的东西会出现在画面上半部分，这一条就能把它挡掉。
        """
        score = float(getattr(detection, "score", 0.0) or 0.0)
        xmin = float(getattr(detection, "xmin", 0.0) or 0.0)
        ymin = float(getattr(detection, "ymin", 0.0) or 0.0)
        xmax = float(getattr(detection, "xmax", 0.0) or 0.0)
        ymax = float(getattr(detection, "ymax", 0.0) or 0.0)
        box_w = xmax - xmin
        box_h = ymax - ymin
        box_min_side = min(box_w, box_h)
        box_long_side = max(box_w, box_h)
        center_y_ratio = (
            0.5 * (ymin + ymax) / image_height if image_height > 0.0 else -1.0
        )
        metrics = {
            "score": score,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "box_min_side": box_min_side,
            "box_long_side": box_long_side,
            "center_y_ratio": center_y_ratio,
        }
        if score < min_score:
            return "score %.2f < %.2f" % (score, min_score), metrics
        if box_long_side < self.trash_min_box_pixels:
            return (
                "box %dx%d 长边 %.0fpx < %.0fpx"
                % (int(box_w), int(box_h), box_long_side, self.trash_min_box_pixels),
                metrics,
            )
        if box_min_side < TRASH_MIN_BOX_SHORT_SIDE:
            return (
                "box %dx%d 细条（短边 %.0fpx < %dpx）"
                % (int(box_w), int(box_h), box_min_side, TRASH_MIN_BOX_SHORT_SIDE),
                metrics,
            )
        if image_height > 0.0 and center_y_ratio < self.trash_min_center_y_ratio:
            return (
                "y_ratio %.3f < %.2f（看着不在画面下半部分，像在桌上）"
                % (center_y_ratio, self.trash_min_center_y_ratio),
                metrics,
            )
        return "", metrics

    def note_trash_detections(self, now):
        """累积本房间看到的垃圾候选（滑动窗口，只数最近 window 秒内的命中）。

        为什么用真滑动窗（保留每次命中的时刻）而不是"间隔超时就清零"：
        后者有个洞 —— 一个误检每 14 秒出现一次，在 15 秒窗口下能慢慢攒到阈值
        并播报。现在是"window 秒内出现 N 次"，既有密度下限，也不要求目标静止：
        车开过去那一两秒里连中几帧就够了。
        """
        cutoff = now - self.trash_observation_window_seconds
        for name in list(self.trash_observations):
            hits = [t for t in self.trash_observations[name]["hits"] if t >= cutoff]
            if hits:
                self.trash_observations[name]["hits"] = hits
            else:
                del self.trash_observations[name]

        with self.lock:
            image = self.latest_image
        image_height = float(image.shape[0]) if image is not None else 0.0
        if image_height <= 0.0 and now - self.trash_no_height_warned_at >= 10.0:
            self.trash_no_height_warned_at = now
            rospy.logwarn(
                "垃圾判据拿不到画面高度（还没有彩色图）；"
                "'是否在地上'那条会被整条跳过"
            )

        if (
            self.trash_detections_topic
            and self.trash_detections_topic != self.detections_topic
        ):
            detections = list(self.trash_detections)
        else:
            with self.lock:
                detections = list(self.latest_detections)
        for detection in detections or []:
            label = str(getattr(detection, "class_name", "") or "").strip().lower()
            if not label:
                continue
            # 先无条件记账（含 person 等所有类名）：这是判断"YOLO 到底看见了
            # 什么"的唯一证据。不记账时，一次 0 命中既可能是房间没放垃圾，
            # 也可能是模型没认出来，两者在现场无法区分。
            self.trash_class_histogram[label] = (
                self.trash_class_histogram.get(label, 0) + 1
            )
            chinese = self.trash_class_names.get(label)
            if not chinese:
                continue
            reject, metrics = self.classify_trash_detection(
                detection, image_height, self.trash_min_score_for(chinese)
            )
            if reject:
                rospy.loginfo_throttle(
                    5.0, "垃圾候选被拒：%s(%s) -> %s", chinese, label, reject
                )
                continue
            entry = self.trash_observations.get(chinese)
            if entry is None:
                entry = {"hits": [], "best_score": 0.0, "label": label}
                self.trash_observations[chinese] = entry
            entry["hits"] = [t for t in entry["hits"] if t >= cutoff]
            entry["hits"].append(now)
            entry["best_score"] = max(entry["best_score"], metrics["score"])
            entry["label"] = label
            if len(entry["hits"]) == self.trash_stability_frames:
                rospy.loginfo(
                    "垃圾候选够稳了：%s(%s) 在 %.0f 秒窗口里命中 %d 次，"
                    "最佳分数 %.2f",
                    chinese,
                    label,
                    self.trash_observation_window_seconds,
                    len(entry["hits"]),
                    entry["best_score"],
                )

    def best_trash_observation(self, now=None):
        """返回最稳的垃圾候选 (中文名, 窗口内命中次数, 最佳分数, 英文类名)。

        已经播报过的那一类直接跳过：否则"刚播报完的瓶子又命中几次"会把
        位置上让不出来，明明旁边还有水果皮够稳，却因为返回了瓶子而播不出去。
        """
        now = time.time() if now is None else now
        cutoff = now - self.trash_observation_window_seconds
        best = None
        for name, data in list(self.trash_observations.items()):
            if name in self.trash_announced:
                continue
            hits = [t for t in data.get("hits", []) if t >= cutoff]
            if len(hits) < self.trash_stability_frames:
                continue
            if best is None or len(hits) > best[1]:
                best = (name, len(hits), data.get("best_score", 0.0), data.get("label", ""))
        return best

    def trash_histogram_summary(self, top=8):
        """本房间 YOLO 实际吐出过的类名（按次数降序），用于现场归因。"""
        items = sorted(self.trash_class_histogram.items(), key=lambda kv: (-kv[1], kv[0]))
        if not items:
            return "(这个房间 YOLO 一个检测框都没吐过)"
        return "、".join("%s x%d" % (name, count) for name, count in items[:top])

    def announce_trash_if_found(self, room_label=""):
        """房间流程走完后看一眼累积结果，有就播报。返回播报的中文名或 None。

        同一类垃圾只说一次（trash_announced）—— 每个房间都说一遍"我找到了垃圾：
        空瓶子"会显得很傻，而且多花播报时间。
        """
        if not self.trash_announce_enabled:
            return None
        best = self.best_trash_observation()
        if best is None:
            rospy.loginfo(
                "%s：这一轮没找到可播报的垃圾（YOLO 见过的类名：%s）",
                room_label or "本房间",
                self.trash_histogram_summary(),
            )
            return None
        chinese, hits, best_score, label = best
        if chinese in self.trash_announced:
            return None
        self.trash_announced.add(chinese)
        speech = self.trash_found_template % chinese
        rospy.loginfo(
            "%s：找到垃圾 %s(%s)，%.0f 秒内命中 %d 次，最佳分数 %.2f -> 播报「%s」",
            room_label or "本房间",
            chinese,
            label,
            self.trash_observation_window_seconds,
            hits,
            best_score,
            speech,
        )
        self.speak(speech, wait=True)
        self.publish_status(
            "trash_found",
            room=room_label,
            trash=chinese,
            label=label,
            observations=hits,
            score=round(float(best_score), 3),
        )
        # 播报过的这一类从账上清掉，别在下一个房间又被算成"新的"
        for name in list(self.trash_observations):
            if name == chinese:
                del self.trash_observations[name]
        return chinese

    # ------------------------------------------------------------------
    # 播报后的实际反应（一）：走近主人
    # ------------------------------------------------------------------
    def approach_pointcloud_callback(self, message):
        """收到一帧点云，只记下来不解析 —— 解析放到真要量距离的时候做。"""
        """收到一帧点云，只记下来不解析 —— 解析放到真要量距离的时候做。"""
        self.approach_pointcloud = message
        self.approach_pointcloud_stamp = time.time()

    def ensure_approach_pointcloud(self):
        """第一次要走近时才订阅点云；返回 True=可以拿到新鲜点云。"""
        if not self.approach_pointcloud_enabled:
            return False
        if self.approach_pointcloud_sub is None:
            try:
                self.approach_pointcloud_sub = rospy.Subscriber(
                    self.points_topic,
                    PointCloud2,
                    self.approach_pointcloud_callback,
                    queue_size=1,
                )
                rospy.loginfo(
                    "走近主人：已订阅点云 %s（用来量真实距离，不占显存）",
                    self.points_topic,
                )
            except Exception as exc:
                rospy.logwarn("订阅点云失败（走近时只能靠画面占比估距离）：%s", exc)
                self.approach_pointcloud_enabled = False
                return False
        # 等第一帧，最多等 2 秒 —— 等不到就先用画面占比退路走着
        deadline = time.time() + 2.0
        while (
            not rospy.is_shutdown()
            and self.approach_pointcloud is None
            and time.time() < deadline
        ):
            rospy.sleep(0.1)
        return self.approach_pointcloud is not None

    def owner_distance_from_pointcloud(self, bbox, image_shape):
        """人体框内取样，返回"人到机器人的**水平**距离"（米）；量不到返回 None。

        取样范围只用框的**上半部分（10%~60% 行）**：
        相机低头装着，人在近处时身前的**地面**会从框的下缘进画面，
        那比人本身还近，取进来会把距离算小、机器人反而停得更远。
        上半部分是躯干，稳稳只有人。

        为什么不是直接取 z：点云在相机光学系里（x 右 / y 下 / z 前），而相机是
        **向下俯约 18°** 装的，z 是"沿光轴"的距离。对躺在地上的人，脚/躯干比相机
        低 ~0.7 米，z 会比水平距离大约 0.2~0.3 米 —— 现场表现就是"人已经很近，
        点云还报 1.4~1.5 米"。这里按俯仰角把每个点投到机器人水平面再取分位数：
            X（前） = z·cosθ - y·sinθ
            Y（侧） = x
            horizontal = sqrt(X² + Y²)
        θ 取 ~approach_camera_pitch_deg（默认 18°，与动作节点标定的 17.8° 一致）。
        """
        cloud = getattr(self, "approach_pointcloud", None)
        stamp = getattr(self, "approach_pointcloud_stamp", None)
        if cloud is None or stamp is None:
            return None
        if time.time() - stamp > self.approach_pointcloud_max_age:
            return None
        if int(getattr(cloud, "height", 0)) <= 1 or int(getattr(cloud, "width", 0)) <= 1:
            return None
        if bbox is None or len(bbox) < 4:
            return None

        # 取样/过滤/投影都在 sample_owner_cloud_points 里（快照式共用同一套），
        # 这里只负责"取分位数"。
        depths = sorted(point[2] for point in self.sample_owner_cloud_points(bbox, image_shape))
        if len(depths) < self.approach_depth_min_samples:
            return None
        index = int(len(depths) * self.approach_depth_percentile / 100.0)
        index = max(0, min(len(depths) - 1, index))
        return depths[index]

    @staticmethod
    def bbox_iou(box_a, box_b):
        """两个像素框 [x1,y1,x2,y2] 的交并比。任一无效返回 0。"""
        if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 1e-6 else 0.0

    def select_approach_target(
        self,
        candidates,
        image_width,
        last_center,
        last_bbox,
        seen_age,
        initial_bbox=None,
        identity_gate=None,
    ):
        """走近时挑目标候选。

        identity_gate=False（摔倒协助用）：**看到人就靠近**，不做身份校验 ——
        房间里不会同时有两位主人，而躺姿框会来回变形，用 IoU 判"是不是同一个人"
        反而会把真主人挡掉（2026-10-10 现场：停在 1.58 米没到位）。
          ① 还没锁定：有 initial_bbox（动作识别那一步给出的主人框）就与它对齐锁定，
             否则挑离画面中心最近的；
          ② 已锁定：挑离上次中心最近的（旧行为）。

        identity_gate=True（默认，挥手求助那条路径）：严格版：
          ① IoU ≥ ~approach_target_min_iou 才算同一个人（取 IoU 最大者）；
          ② 上一帧目标刚漏检（≤ ~approach_relock_grace_seconds）且中心很近时允许续上；
          ③ 没锁定时同上；④ 其余返回 None（原地等，不跟别人走）。
        """
        if not candidates:
            return None
        use_gate = (
            self.approach_identity_gate if identity_gate is None else bool(identity_gate)
        )
        if image_width is None:
            return candidates[0]

        def nearest_to(center):
            return min(
                candidates,
                key=lambda item: abs(
                    self.candidate_center_norm(item, image_width) - center
                ),
            )

        if last_bbox is None:
            # 第一帧：优先用"主人自己的框"对齐（动作识别那一步给的），其次画面中心
            if initial_bbox is not None and len(initial_bbox) >= 4:
                anchor = 0.5 * (float(initial_bbox[0]) + float(initial_bbox[2])) / max(
                    1.0, float(image_width)
                )
                best = None
                best_iou = 0.0
                for item in candidates:
                    iou = self.bbox_iou(initial_bbox, item.get("bbox"))
                    if iou > best_iou:
                        best, best_iou = item, iou
                if best is not None and best_iou > 0.05:
                    return best
                return nearest_to(anchor)
            return nearest_to(0.5)

        if not use_gate:
            # 最简行为：看到人就靠近（只有一位主人，不需要身份校验）
            return nearest_to(last_center)

        best = None
        best_iou = 0.0
        for item in candidates:
            iou = self.bbox_iou(last_bbox, item.get("bbox"))
            if iou > best_iou:
                best, best_iou = item, iou
        if best is not None and best_iou >= self.approach_target_min_iou:
            return best
        if seen_age <= self.approach_relock_grace_seconds:
            near = nearest_to(last_center)
            center = self.candidate_center_norm(near, image_width)
            if abs(center - last_center) <= self.approach_relock_center_tolerance:
                return near
        return None

    def approach_owner(
        self,
        owner_name="",
        standoff_distance=None,
        min_distance=None,
        initial_bbox=None,
        lost_seconds=None,
        identity_gate=None,
    ):
        """走到主人跟前。返回 True=到位，False=没走到（人会走开、超时等）。

        用的是"边看边走"的闭环：每一轮重新检测一次主人，
          框偏左/偏右 -> 原地转一点；框基本居中 -> 往前开一点；
          点云量出的距离到 ~approach_standoff_distance 米 -> 到了，停。
        量不到点云时退回"框高占比"（不精确，日志里会写明是估计值）。
        主人从画面里消失超过 ~approach_lost_seconds 秒就立刻停 ——
        宁可没走到，也不能照着一个已经不在了的位置盲开。

        standoff_distance：这一次要走到多少米。默认用 ~approach_standoff_distance；
        摔倒协助会传一个更近的值（机械臂要够得着人）。
        min_distance：这一次的安全下限（比它更近就绝不再往前）。默认用
        ~approach_min_distance；摔倒协助会一起传更低的下限，否则 0.5 米的目标
        会被通用的 0.6 米下限挡住。

        initial_bbox：动作识别那一步给出的**主人人体框**（像素）。用它做第一帧的
        锁定锚点，比"离画面中心最近"可靠得多（同一画面里还有别人时尤其明显）。

        lost_seconds：这一次"连续多久看不到人就停"。默认 ~approach_lost_seconds；
        摔倒协助会传更长的值（躺姿的人体框本身会闪断，2 秒太短）。

        identity_gate：这一次要不要做"是不是同一个人"的严格校验。默认沿用
        ~approach_identity_gate；**摔倒协助传 False**（房间里不会同时有两位主人，
        看到人就靠近）。
        """
        if not self.approach_enabled:
            rospy.loginfo("走近主人已关闭（approach_enabled=false）")
            return False
        target_distance = (
            float(self.approach_standoff_distance)
            if standoff_distance is None
            else max(0.3, float(standoff_distance))
        )
        floor_distance = (
            float(self.approach_min_distance)
            if min_distance is None
            else max(0.2, float(min_distance))
        )
        # 下限不能高于目标，否则永远停不到目标距离
        floor_distance = min(floor_distance, target_distance)
        # 这一次"看不到人多长时间就中止"：躺姿框会闪断，摔倒协助会传更长的值
        lost_limit = max(
            0.5,
            float(self.approach_lost_seconds if lost_seconds is None else lost_seconds),
        )

        target_label = owner_name or "主人"
        has_pointcloud = self.ensure_approach_pointcloud()
        if str(self.approach_start_text or "").strip():
            # 同一次靠近只播一次（摔倒三级回退 / 挥手走近都适用）；
            # 内部是 wait=False，和"开始走过去"并行，不额外花时间。
            self.fall_speak_approach_start_once()
        rospy.loginfo(
            "开始走近%s：目标距离 %.2f 米（点云=%s），安全下限 %.2f 米，"
            "前进速度 %.2f m/s，最多 %.0f 秒",
            target_label,
            target_distance,
            "可用" if has_pointcloud else "不可用，退回画面占比 %.2f" % self.approach_standoff_ratio,
            floor_distance,
            self.approach_forward_speed,
            self.approach_max_seconds,
        )

        deadline = time.time() + self.approach_max_seconds
        last_seen_time = time.time()
        last_center = 0.5
        last_ratio = 0.0
        last_distance = None
        last_target_bbox = None
        last_target_seen = 0.0
        rate = rospy.Rate(10)
        result = False
        try:
            while not rospy.is_shutdown() and time.time() < deadline:
                image, detections = self.snapshot()
                candidates = (
                    self.person_candidates(image, detections)
                    if image is not None
                    else []
                )
                # 认"还是刚才那一位"：先看框的重叠（IoU），不再只看"离中心最近"，
                # 否则画面里走过别人时目标会被抢走（采样框跟着别人跑）。
                candidate = self.select_approach_target(
                    candidates,
                    image.shape[1] if image is not None else None,
                    last_center,
                    last_target_bbox,
                    time.time() - last_target_seen if last_target_seen else 1e9,
                    initial_bbox=initial_bbox,
                    identity_gate=identity_gate,
                )
                if candidate is not None:
                    last_target_bbox = candidate.get("bbox", last_target_bbox)
                    last_target_seen = time.time()

                if candidate is None:
                    # 看不到人（画面里没人 / 画面本身断了）：站着别动，等一小会儿；
                    # 超过 ~approach_lost_seconds 就收工 —— 绝不照着旧位置盲开。
                    self.stop_base()
                    if candidates:
                        rospy.logwarn_throttle(
                            2.0,
                            "走近%s：画面里有 %d 个人，但没有一个是刚才那位"
                            "（疑似被别人遮挡/目标走开），原地等",
                            target_label,
                            len(candidates),
                        )
                    if time.time() - last_seen_time >= lost_limit:
                        rospy.logwarn(
                            "走近%s中止：连续 %.1f 秒看不到人，停下（不盲开）",
                            target_label,
                            time.time() - last_seen_time,
                        )
                        break
                    rate.sleep()
                    continue

                last_seen_time = time.time()
                image_height = max(1.0, float(image.shape[0]))
                bbox = candidate.get("bbox", [0, 0, 0, 0])
                box_height = max(0.0, float(bbox[3]) - float(bbox[1]))
                last_ratio = box_height / image_height
                last_center = self.candidate_center_norm(candidate, image.shape[1])
                x_error = last_center - 0.5

                # 优先用点云量真实距离；量不到才退回"框高占比"。
                last_distance = self.owner_distance_from_pointcloud(bbox, image.shape)
                if last_distance is not None:
                    rospy.loginfo_throttle(
                        1.5,
                        "走近%s：点云距离 %.2f 米（目标 %.2f，安全下限 %.2f）",
                        target_label,
                        last_distance,
                        target_distance,
                        floor_distance,
                    )
                    if last_distance <= target_distance:
                        self.stop_base()
                        rospy.loginfo(
                            "已经走到%s跟前：点云距离 %.2f 米 ≤ 目标 %.2f 米，停",
                            target_label,
                            last_distance,
                            target_distance,
                        )
                        result = True
                        break
                    if last_distance <= floor_distance:
                        # 安全底线：再怎么也不往前发了
                        self.stop_base()
                        rospy.logwarn(
                            "距离%s只剩 %.2f 米（安全下限 %.2f），不再前进",
                            target_label,
                            last_distance,
                            floor_distance,
                        )
                        result = True
                        break
                elif last_ratio >= self.approach_standoff_ratio:
                    self.stop_base()
                    rospy.loginfo(
                        "已经走到%s跟前：量不到点云，按框高占比 %.2f（目标 %.2f）判到位，停",
                        target_label,
                        last_ratio,
                        self.approach_standoff_ratio,
                    )
                    result = True
                    break

                twist = Twist()
                if abs(x_error) > self.approach_center_tolerance:
                    # 没对准就先转，不对着人开
                    twist.angular.z = max(
                        -self.approach_turn_speed,
                        min(self.approach_turn_speed, -1.6 * x_error),
                    )
                    self.update_yolo_window("走近主人：对准")
                else:
                    twist.linear.x = self.approach_forward_speed
                    twist.angular.z = max(
                        -self.approach_turn_speed,
                        min(self.approach_turn_speed, -0.8 * x_error),
                    )
                    self.update_yolo_window("走近主人：前进")
                self.cmd_pub.publish(twist)
                rate.sleep()
        finally:
            self.stop_base()

        if not result:
            rospy.logwarn(
                "没走到%s跟前（最后距离 %s / 目标 %.2f 米），就地继续后面的流程",
                target_label,
                (
                    "%.2f 米" % last_distance
                    if last_distance is not None
                    else "量不到，框高占比 %.2f" % last_ratio
                ),
                target_distance,
            )
        self.publish_status(
            "owner_approach_finished",
            owner_name=owner_name,
            arrived=bool(result),
            box_ratio=round(last_ratio, 3),
            distance=None if last_distance is None else round(last_distance, 3),
        )
        return result

    # ------------------------------------------------------------------
    # 播报后的实际反应（二）：挥手求助对话
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 摔倒靠近：快照式（方案 B 第一步）
    # ------------------------------------------------------------------
    @staticmethod
    def wrap_angle(value):
        """把角度规整到 [-pi, pi]。"""
        return math.atan2(math.sin(value), math.cos(value))

    def scan_callback(self, message):
        """记两个数：正前方最近距离（爬行/直行的硬护栏）+ 全向最近距离（旋转护栏）。

        全向那个是 2026-10-16 加的：车体原地旋转时扫到的是**侧面/斜后方**的东西
        （现场撞过椅子腿），正前方 ±25° 那条护栏完全管不到。
        """
        half = math.radians(self.scan_forward_half_angle_deg)
        range_min = max(0.05, float(getattr(message, "range_min", 0.0) or 0.0))
        range_max = min(3.0, float(getattr(message, "range_max", 0.0) or 3.0))
        best = None
        best_any = None
        for index, raw in enumerate(getattr(message, "ranges", []) or []):
            try:
                distance = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance):
                continue
            if distance < range_min or distance > range_max:
                continue
            if best_any is None or distance < best_any:
                best_any = distance
            angle = float(message.angle_min) + index * float(message.angle_increment)
            if abs(math.atan2(math.sin(angle), math.cos(angle))) > half:
                continue
            if best is None or distance < best:
                best = distance
        with self.lock:
            self._scan_forward_min = best
            self._scan_any_min = best_any
            self._scan_stamp = time.time()
            self._scan_msg_stamp = self._scan_stamp

    def ensure_scan_subscription(self):
        if self._scan_sub is not None:
            return True
        try:
            from sensor_msgs.msg import LaserScan

            self._scan_sub = rospy.Subscriber(
                self.scan_topic, LaserScan, self.scan_callback, queue_size=1
            )
            rospy.loginfo(
                "已订阅 %s（硬件自检看数据、摔倒靠近当正前方护栏）", self.scan_topic
            )
            return True
        except Exception as exc:  # noqa: BLE001 - 没护栏也要能跑
            rospy.logwarn("订阅 %s 失败（前向护栏不可用）：%s", self.scan_topic, exc)
            return False

    def scan_forward_min_distance(self):
        """正前方最小距离（米）；超过 1 秒没更新或没数据返回 None。"""
        with self.lock:
            if self._scan_forward_min is None:
                return None
            if time.time() - float(self._scan_stamp or 0.0) > 1.0:
                return None
            return float(self._scan_forward_min)

    def scan_any_min_distance(self):
        """**全向**（360°）最小距离（米）；超过 1 秒没更新或没数据返回 None。

        原地旋转的护栏用这个：正前方那条 `scan_forward_min_distance` 只有 ±25°，
        车体旋转时会横扫侧面，必须看全向。
        """
        with self.lock:
            if self._scan_any_min is None:
                return None
            if time.time() - float(self._scan_stamp or 0.0) > 1.0:
                return None
            return float(self._scan_any_min)

    def estimate_cloud_floor_line(self, cloud, band_count=6):
        """从点云画面下半部拟合地面线 y = a·z + b（相机光学系），拿不到返回 None。

        做法与动作节点一致：把画面 55%~98% 行分 6 条带，每带取"最靠下（y 最大）"
        的那批点当地面样本，再最小二乘拟合。摔倒路径用它把"贴地点"剔掉，只留身体。
        """
        if cloud is None:
            return None
        cloud_height = int(getattr(cloud, "height", 0))
        cloud_width = int(getattr(cloud, "width", 0))
        if cloud_height <= 1 or cloud_width <= 1:
            return None
        v_start = self.clamp_value(int(cloud_height * 0.55), 0, cloud_height - 1)
        v_end = max(v_start + 1, self.clamp_value(int(cloud_height * 0.98), 0, cloud_height))
        u1 = self.clamp_value(int(cloud_width * 0.15), 0, cloud_width - 1)
        u2 = max(u1 + 1, self.clamp_value(int(cloud_width * 0.85), 0, cloud_width))
        samples = []
        for index in range(max(1, int(band_count))):
            b1 = v_start + int(round((v_end - v_start) * index / float(band_count)))
            b2 = v_start + int(round((v_end - v_start) * (index + 1) / float(band_count)))
            uvs = [
                (u, v)
                for v in range(b1, max(b1 + 1, b2), 2)
                for u in range(u1, u2, 8)
            ]
            if not uvs:
                continue
            try:
                points = [
                    (float(x), float(y), float(z))
                    for x, y, z in pc2.read_points(
                        cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs
                    )
                    if math.isfinite(float(z)) and math.isfinite(float(y))
                    and 0.2 < float(z) < 6.0
                ]
            except Exception:  # noqa: BLE001
                continue
            if len(points) < 10:
                continue
            points.sort(key=lambda item: item[1])
            lowest = points[-max(3, len(points) // 10):]
            z_med = sorted(item[2] for item in lowest)[len(lowest) // 2]
            y_med = sorted(item[1] for item in lowest)[len(lowest) // 2]
            samples.append((z_med, y_med))
        if len(samples) < 3:
            return None
        n = float(len(samples))
        sum_z = sum(item[0] for item in samples)
        sum_y = sum(item[1] for item in samples)
        sum_zz = sum(item[0] * item[0] for item in samples)
        sum_zy = sum(item[0] * item[1] for item in samples)
        denom = n * sum_zz - sum_z * sum_z
        if abs(denom) < 1e-9:
            return None
        slope = (n * sum_zy - sum_z * sum_y) / denom
        intercept = (sum_y - slope * sum_z) / n
        return slope, intercept

    def sample_owner_cloud_points(
        self,
        bbox,
        image_shape,
        band=None,
        floor_filter=False,
        anchor=None,
        depth_band=None,
    ):
        """人体框上半部分 → [(forward, lateral, horizontal)]（已按相机俯仰角投影）。

        两种取样方式（2026-10-11 加 anchor 一路）：
        · anchor=(u,v) 给了（推荐，摔倒路径走这条）：只取锚点 ±fall_anchor_radius_px
          的小窗口，再用"深度一致性"剔掉窗口里落在人**后面**的家具/墙。
          锚点=左右肩+左右髋中点（见 torso_anchor_from_candidate）。
          为什么：现场 RViz 实测，整条身体带里"大部分点在家具/墙面上"，中位数被
          拉到远端髋骨，机器人于是朝头/脚走。窗口点数太少会自动退回下面的整条身体带。
        · anchor 没给（默认）：整条身体带，与 owner_distance_from_pointcloud 共用同一套
          取样/过滤：
        · band 指定取框高的哪一段（默认 10%~60%：避开低头相机照进来的地面；
          摔倒路径传 15%~85% 取**整条身体**，再靠 floor_filter 剔掉贴地点）；
        · 每 4 像素取一个点；
        · 点云 z（光轴方向）按 ~approach_camera_pitch_deg 投影到机器人水平面：
              forward = z·cosθ − y·sinθ      lateral = x
        """
        cloud = getattr(self, "approach_pointcloud", None)
        stamp = getattr(self, "approach_pointcloud_stamp", None)
        if cloud is None or stamp is None:
            return []
        if time.time() - stamp > self.approach_pointcloud_max_age:
            return []
        if int(getattr(cloud, "height", 0)) <= 1 or int(getattr(cloud, "width", 0)) <= 1:
            return []
        if bbox is None or len(bbox) < 4:
            return []

        image_height, image_width = image_shape[:2]
        cloud_width = int(cloud.width)
        cloud_height = int(cloud.height)
        scale_x = float(cloud_width) / max(1.0, float(image_width))
        scale_y = float(cloud_height) / max(1.0, float(image_height))

        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        box_height = max(2.0, y2 - y1)
        band_low, band_high = (0.10, 0.60) if band is None else (band[0], band[1])

        pitch = math.radians(self.approach_camera_pitch_deg)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        floor_line = None
        if floor_filter and getattr(self, "fall_floor_filter_enabled", True):
            floor_line = self.estimate_cloud_floor_line(cloud)
        clearance = float(getattr(self, "fall_floor_clearance", 0.12))
        floor_scale = 1.0
        if floor_line is not None:
            floor_scale = math.sqrt(1.0 + float(floor_line[0]) ** 2)

        def build_uvs(px1, py1, px2, py2):
            sample_x1 = self.clamp_value(int(px1 * scale_x), 0, cloud_width - 1)
            sample_x2 = self.clamp_value(int(px2 * scale_x), sample_x1 + 1, cloud_width)
            sample_y1 = self.clamp_value(int(py1 * scale_y), 0, cloud_height - 1)
            sample_y2 = self.clamp_value(int(py2 * scale_y), sample_y1 + 1, cloud_height)
            return [
                (u, v)
                for v in range(sample_y1, sample_y2, 4)
                for u in range(sample_x1, sample_x2, 4)
            ]

        def read_window(uvs):
            """读一个像素窗口里的点，顺手剔掉太近/太远和贴地的点。"""
            if not uvs:
                return []
            try:
                raw_points = list(
                    pc2.read_points(
                        cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn_throttle(5.0, "读点云失败：%s", exc)
                return []
            kept = []
            for x, y, z in raw_points:
                x = float(x)
                y = float(y)
                z = float(z)
                if not (0.15 < z < 8.0):
                    continue
                if floor_line is not None:
                    distance_to_floor = (
                        abs(y - (floor_line[0] * z + floor_line[1])) / floor_scale
                    )
                    if distance_to_floor < clearance:
                        continue
                kept.append((x, y, z))
            return kept

        def project(points):
            sampled = []
            for x, y, z in points:
                forward = z * cos_p - y * sin_p
                sampled.append((forward, x, math.hypot(forward, x)))
            return sampled

        # ① 躯干锚点窗口（首选）：锚点=左右肩+左右髋中点，只在这个小窗口里取点，
        #    再按深度一致性剔掉窗口里落在人后面的家具/墙（|z−窗口中位深度| ≤ depth_window）。
        if anchor is not None:
            radius = self.fall_anchor_radius_px
            window_points = read_window(
                build_uvs(
                    float(anchor[0]) - radius,
                    float(anchor[1]) - radius,
                    float(anchor[0]) + radius,
                    float(anchor[1]) + radius,
                )
            )
            depth_window = (
                self.fall_anchor_depth_band
                if depth_band is None
                else max(0.15, float(depth_band))
            )
            if window_points:
                depths = sorted(point[2] for point in window_points)
                reference = depths[len(depths) // 2]
                kept = [
                    point
                    for point in window_points
                    if abs(point[2] - reference) <= depth_window
                ]
                if len(kept) >= min(len(window_points), 5):
                    window_points = kept
            if len(window_points) >= self.fall_anchor_min_points:
                self._fall_last_sample_source = "anchor"
                return project(window_points)
            rospy.loginfo_throttle(
                5.0,
                "躯干锚点窗口只有 %d 个点（需 ≥%d），退回整条身体带采样",
                len(window_points),
                self.fall_anchor_min_points,
            )

        # ② 退回整条身体带（原行为，未改动）
        self._fall_last_sample_source = "band"
        return project(
            read_window(
                build_uvs(
                    x1,
                    y1 + box_height * band_low,
                    x2,
                    y1 + box_height * band_high,
                )
            )
        )

    def owner_position_from_pointcloud(
        self,
        bbox,
        image_shape,
        band=None,
        floor_filter=False,
        min_points=None,
        anchor=None,
    ):
        """主人相对机器人的水平位置（中位数），拿不到返回 None。

        返回 dict：forward（前）/ lateral（侧）/ distance / bearing / samples / bbox。
        取中位数而不是低分位数：快照用于"朝哪走"，中位数对背景点更稳
        （参考实现 server_robot3 的 capture_fallen_owner_position 也是中位数）。
        anchor=(u,v) 时走"躯干锚点窗口"采样（见 sample_owner_cloud_points），
        返回值里的 source 会告诉你这次用的是 anchor 还是 band。
        """
        sampled = self.sample_owner_cloud_points(
            bbox, image_shape, band=band, floor_filter=floor_filter, anchor=anchor
        )
        need = (
            self.fall_snapshot_min_points if min_points is None else max(1, int(min_points))
        )
        if len(sampled) < need:
            return None
        forwards = sorted(point[0] for point in sampled)
        laterals = sorted(point[1] for point in sampled)
        distances = sorted(point[2] for point in sampled)
        forward = forwards[len(forwards) // 2]
        lateral = laterals[len(laterals) // 2]
        distance = distances[len(distances) // 2]
        # 调试可视化用：把这次参与估计的点（前/侧，机器人系）带回去。
        # 最多 ~240 个，够在 RViz 里看出"这坨点到底是人还是家具"。
        step = max(1, len(sampled) // 240)
        points_xy = [
            (float(point[0]), float(point[1])) for point in sampled[::step]
        ]
        return {
            "forward": forward,
            "lateral": lateral,
            "distance": distance,
            "bearing": math.atan2(lateral, max(0.20, forward)),
            "samples": len(sampled),
            "points_xy": points_xy,
            "source": getattr(self, "_fall_last_sample_source", "band"),
            "bbox": [float(value) for value in bbox[:4]],
        }

    def fall_tracking_candidate(self, candidates, ref_bbox):
        """两人场景防串：只认与"主人框"仍重叠、且面积没差太多的候选。

        返回候选 dict，或 None（= 没找到"还是那位主人"的框）。
        为什么要加面积闸门：躺着的人框又宽又扁、站着的人框接近方形；画面里第二个人
        走进来时，只按"离中心最近"或单纯 IoU 很容易串过去（实测采样框串到站着的人，
        紧接着"量不到人"被误判成"人已掉出画面"，触发盲走）。
        """
        if not candidates:
            return None
        if ref_bbox is None or len(ref_bbox) < 4:
            return candidates[0]
        ref_area = max(
            1.0,
            (float(ref_bbox[2]) - float(ref_bbox[0]))
            * (float(ref_bbox[3]) - float(ref_bbox[1])),
        )
        best = None
        best_iou = 0.0
        for item in candidates:
            box = item.get("bbox")
            if not box or len(box) < 4:
                continue
            iou = self.bbox_iou(ref_bbox, box)
            if iou < self.fall_tracking_min_iou:
                continue
            area = max(
                1.0,
                (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])),
            )
            ratio = max(ref_area / area, area / ref_area)
            if ratio > self.fall_tracking_max_area_ratio:
                continue
            if iou > best_iou:
                best, best_iou = item, iou
        return best

    # COCO-17 关键点下标：左右肩(5,6) + 左右髋(11,12) = 躯干
    TORSO_KEYPOINTS = (5, 6, 11, 12)

    def body_cloud_points(self, bbox, image_shape, band=None, floor_filter=False, limit=240):
        """整条身体带的采样点（不锚定），只用来估"头↔脚"主方向。

        锚点窗口只有躯干那一小块，估不出身体方向；这里再采一次整条带
        （一次点云读取，几毫秒），结果**不参与位置估计**，只决定"从哪一侧接近"。
        注意：会覆盖 self._fall_last_sample_source，所以必须在位置估计之后调用。
        """
        sampled = self.sample_owner_cloud_points(
            bbox, image_shape, band=band, floor_filter=floor_filter, anchor=None
        )
        if not sampled:
            return []
        step = max(1, len(sampled) // max(1, int(limit)))
        return [(float(point[0]), float(point[1])) for point in sampled[::step]]

    def body_axis_from_points(self, points, center):
        """用整条身体带的点估"头↔脚"主方向：返回 (ux, uy, spread) 或 None。

        做法：先把离身体中心太远的点（墙/家具残留）剔掉，再算 2x2 协方差矩阵的
        最大特征向量——2x2 有解析解，不引 numpy：
            θ = 0.5 * atan2(2·sxy, sxx − syy)
        spread ≈ 身体半长，太小说明"这坨点不成条"（多半只圈到一部分），返回 None。
        """
        if not points or center is None:
            return None
        radius = self.fall_body_axis_max_radius
        cx, cy = float(center[0]), float(center[1])
        kept = [
            (float(point[0]), float(point[1]))
            for point in points
            if math.hypot(float(point[0]) - cx, float(point[1]) - cy) <= radius
        ]
        if len(kept) < 20:
            return None
        count = float(len(kept))
        mean_x = sum(point[0] for point in kept) / count
        mean_y = sum(point[1] for point in kept) / count
        sxx = sum((point[0] - mean_x) ** 2 for point in kept) / count
        syy = sum((point[1] - mean_y) ** 2 for point in kept) / count
        sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in kept) / count
        theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        # 两个特征值（2x2 解析解），取大的那个算半长
        trace = sxx + syy
        diff = math.sqrt(max(0.0, (sxx - syy) ** 2 + 4.0 * sxy * sxy))
        spread = math.sqrt(max(0.0, (trace + diff) / 2.0))
        if spread < self.fall_body_axis_min_spread:
            return None
        return (math.cos(theta), math.sin(theta), spread)

    def body_obb_from_points(self, points, center):
        """身体点 → 地面上的"旋转矩形"（OBB）：(cx, cy, ux, uy, half_len, half_wid)。

        RViz 里用它画人体的**大致形状和位置**，比一堆散点直观得多：
        长边=头↔脚方向，短边=身体宽度。点数太少或不成条返回 None。
        """
        axis = self.body_axis_from_points(points, center)
        if axis is None:
            return None
        ux, uy = axis[0], axis[1]
        px, py = -uy, ux
        cx, cy = float(center[0]), float(center[1])
        radius = self.fall_body_axis_max_radius
        kept = [
            (float(point[0]), float(point[1]))
            for point in points
            if math.hypot(float(point[0]) - cx, float(point[1]) - cy) <= radius
        ]
        if len(kept) < 20:
            return None
        along = [(point[0] - cx) * ux + (point[1] - cy) * uy for point in kept]
        across = [(point[0] - cx) * px + (point[1] - cy) * py for point in kept]
        along.sort()
        across.sort()
        # 取 5%~95% 分位当边长：避免一两个飞点把矩形撑大
        def pct(values, ratio):
            index = max(0, min(len(values) - 1, int(round((len(values) - 1) * ratio))))
            return values[index]

        low = pct(along, 0.05)
        high = pct(along, 0.95)
        left = pct(across, 0.05)
        right = pct(across, 0.95)
        centre_along = 0.5 * (low + high)
        centre_across = 0.5 * (left + right)
        half_len = max(0.10, 0.5 * (high - low))
        half_wid = max(0.08, 0.5 * (right - left))
        return (
            cx + ux * centre_along + px * centre_across,
            cy + uy * centre_along + py * centre_across,
            ux,
            uy,
            half_len,
            half_wid,
        )

    def fall_standoff_goal_base(
        self, owner_x, owner_y, standoff, body_xy=None, label="主人"
    ):
        """算"机器人系里"的 standoff 目标点 + 期望朝向。返回 (gx, gy, face_yaw, info)。

        · 默认（没有身体方向 / 关掉）：沿"主人→机器人"连线退 standoff（原行为）；
        · 有身体主方向且打开侧向接近：**退在身体侧面**，光标朝身体中心 ——
          这样机器人不会从头或脚的方向怼过去。
        · 无论哪种，目标点都必须离任何身体点 ≥ fall_nav_body_clearance，
          不够就把 standoff 往外推（最多 +fall_nav_side_extend_max）。
        """
        info = {"mode": "line", "spread": None, "min_body_distance": None}
        norm = max(0.05, math.hypot(owner_x, owner_y))
        unit_x, unit_y = owner_x / norm, owner_y / norm
        approach = (-unit_x, -unit_y)  # 从主人指向机器人（默认）
        axis = None
        if self.fall_body_axis_enabled and self.fall_side_approach_enabled:
            axis = self.body_axis_from_points(body_xy, (owner_x, owner_y))
        if axis is not None:
            perp = (-axis[1], axis[0])
            # 取"朝机器人这一侧"的那个垂直方向
            if perp[0] * approach[0] + perp[1] * approach[1] < 0.0:
                perp = (-perp[0], -perp[1])
            approach = perp
            info["mode"] = "side"
            info["spread"] = float(axis[2])
            info["axis"] = (float(axis[0]), float(axis[1]))
        effective = max(0.35, float(standoff))
        goal_x = float(owner_x) + approach[0] * effective
        goal_y = float(owner_y) + approach[1] * effective
        # 身体安全距离校验：不够就把目标往外推
        if body_xy:
            limit = effective + self.fall_nav_side_extend_max
            while True:
                nearest = min(
                    math.hypot(goal_x - float(point[0]), goal_y - float(point[1]))
                    for point in body_xy
                )
                info["min_body_distance"] = float(nearest)
                if nearest >= self.fall_nav_body_clearance or effective >= limit:
                    break
                effective = min(limit, effective + 0.10)
                goal_x = float(owner_x) + approach[0] * effective
                goal_y = float(owner_y) + approach[1] * effective
        info["standoff"] = effective
        # 身体的大致形状（旋转矩形），给 RViz 画图 + 现场判断用
        if body_xy:
            info["obb"] = self.body_obb_from_points(body_xy, (owner_x, owner_y))
        # 朝向：从目标点看向身体中心
        face_x = float(owner_x) - goal_x
        face_y = float(owner_y) - goal_y
        face_yaw = math.atan2(face_y, face_x)
        return goal_x, goal_y, face_yaw, info

    def torso_anchor_from_candidate(self, candidate):
        """从姿态候选里取"躯干中心"的图像像素坐标；拿不到返回 None。

        为什么用躯干而不是"整条身体带"：躺姿的检测框会把家具/墙面圈进来，
        整条带的中位数会被拉到身体的某一端（实测是远端髋骨）。躯干中点才是
        机器人该停在 0.5 m 外的那个位置。
        置信度不足的关键点（< ~fall_torso_keypoint_min_conf）直接丢掉，
        只剩单侧肩/髋也能算；一个都不剩就返回 None，调用方退回原采样。
        """
        if not getattr(self, "fall_torso_anchor_enabled", True) or not candidate:
            return None
        points = candidate.get("keypoints")
        # ⚠️ 这里绝不能用 `if not points:` —— ultralytics 给的是 numpy 数组，
        #    `not ndarray` 会抛 "truth value of an array is ambiguous"，
        #    现场后果就是：摔倒播报完之后整个靠近流程被异常打断（10-11 21:36 那次）。
        if points is None:
            return None
        try:
            point_count = len(points)
        except TypeError:
            return None
        if point_count <= 0:
            return None
        confidences = candidate.get("keypoint_confidences")
        picked = []
        for index in self.TORSO_KEYPOINTS:
            if index >= len(points):
                continue
            point = points[index]
            if point is None:
                continue
            try:
                if len(point) < 2:
                    continue
            except TypeError:  # 标量/0维数组：按"没检出来"处理
                continue
            confidence = 1.0
            if confidences is not None and index < len(confidences):
                try:
                    confidence = float(confidences[index])
                except (TypeError, ValueError):
                    confidence = 1.0
            if confidence < self.fall_torso_keypoint_min_conf:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if x <= 0.0 and y <= 0.0:
                # 上游"没检出来"的关键点就是 (0,0)，别当成画面左上角
                continue
            picked.append((x, y))
        if not picked:
            return None
        return (
            sum(point[0] for point in picked) / float(len(picked)),
            sum(point[1] for point in picked) / float(len(picked)),
        )

    # ------------------------------------------------------------------
    # 摔倒靠近：RViz 调试可视化（只发 Marker，不参与任何决策）
    # ------------------------------------------------------------------
    # 为什么要有它：现场"机器人朝头/脚走""好像往家具走"这类问题，光看日志
    # 判断不出来——必须把"用来算位置的那些点"和"算出来的目标点"直接画在
    # 点云图上。RViz → Add → MarkerArray → 选 /fall_approach/markers 即可。
    def _fall_debug_publisher(self):
        """按需创建 MarkerArray 发布器（latched）；不可用返回 None。"""
        if not self.fall_debug_markers:
            return None
        if Marker is None or MarkerArray is None:
            if not self._fall_debug_warned:
                rospy.logwarn("visualization_msgs 不可用，摔倒靠近调试可视化关闭")
                self._fall_debug_warned = True
                self.fall_debug_markers = False
            return None
        if self._fall_debug_pub is None:
            self._fall_debug_pub = rospy.Publisher(
                self.fall_debug_marker_topic, MarkerArray, queue_size=2, latch=True
            )
            rospy.loginfo(
                "摔倒靠近调试可视化已开启：%s"
                "（RViz → Add → MarkerArray，Fixed Frame 用 %s；"
                "不要了就加 _fall_debug_markers:=false）",
                self.fall_debug_marker_topic,
                self.fall_nav_frame,
            )
        return self._fall_debug_pub

    @staticmethod
    def _fall_debug_base_marker(marker_id, frame_id, marker_type):
        marker = Marker()
        marker.header.frame_id = frame_id
        try:
            marker.header.stamp = rospy.Time.now()
        except Exception:  # noqa: BLE001 - 未 init_node（离线测试）时用 0=取最新 TF
            marker.header.stamp = rospy.Time(0)
        marker.ns = "fall_approach"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0.0)
        return marker

    @staticmethod
    def _fall_debug_set_color(marker, r, g, b, a=0.9):
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

    def _fall_debug_point_marker(self, marker_id, frame_id, x, y, z, rgba, size):
        marker = self._fall_debug_base_marker(marker_id, frame_id, Marker.SPHERE)
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.scale.x = marker.scale.y = marker.scale.z = float(size)
        self._fall_debug_set_color(marker, *rgba)
        return marker

    def _fall_debug_line_marker(self, marker_id, frame_id, points, rgba, width=0.02):
        marker = self._fall_debug_base_marker(marker_id, frame_id, Marker.LINE_STRIP)
        marker.scale.x = float(width)
        self._fall_debug_set_color(marker, *rgba)
        marker.points = [
            Point(float(px), float(py), float(pz)) for px, py, pz in points
        ]
        return marker

    def _fall_debug_points_marker(self, marker_id, frame_id, points, rgba, size=0.02):
        marker = self._fall_debug_base_marker(marker_id, frame_id, Marker.POINTS)
        marker.scale.x = marker.scale.y = float(size)
        self._fall_debug_set_color(marker, *rgba)
        marker.points = [
            Point(float(px), float(py), float(pz)) for px, py, pz in points
        ]
        return marker

    def _fall_debug_text_marker(self, marker_id, frame_id, x, y, z, text, rgba):
        marker = self._fall_debug_base_marker(
            marker_id, frame_id, Marker.TEXT_VIEW_FACING
        )
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.scale.z = 0.16
        marker.text = str(text)
        self._fall_debug_set_color(marker, rgba[0], rgba[1], rgba[2], 1.0)
        return marker

    def publish_fall_debug_markers(
        self,
        label,
        owner_base=None,
        samples_base=None,
        body_base=None,
        axis_base=None,
        obb_base=None,
        standoff_base=None,
        owner_map=None,
        goal_map=None,
        pose_map=None,
        owner_distance=None,
        standoff_distance=None,
        note="",
    ):
        """把摔倒靠近用到的关键点画到 RViz（只画，不决策）。

        坐标系：能拿到 map 位姿时，"主人估计点 / standoff 目标点 / 机器人位置"
        都画在 map 系（与真实房间对齐，方向不对一眼就能看出来）；点云采样点画在
        base_footprint，RViz 用 TF 自动摆正。
        颜色：红=主人估计点（躯干锚点算出来的），绿=standoff 目标点，蓝=机器人，
              青=**这次真正用来估位置的点**（躯干锚点窗口，只有一小撮，正常），
              灰=整条身体带的点（看身体轮廓用），黄线=身体主方向（头↔脚），
              亮绿矩形=**人体大致形状**（旋转矩形：长边头↔脚、短边身体宽度），
              橙线=机器人→主人，绿线=主人→目标。
        """
        try:
            publisher = self._fall_debug_publisher()
            if publisher is None:
                return None
            return self._publish_fall_debug_markers(
                publisher,
                label,
                owner_base,
                samples_base,
                body_base,
                axis_base,
                obb_base,
                standoff_base,
                owner_map,
                goal_map,
                pose_map,
                owner_distance,
                standoff_distance,
                note,
            )
        except Exception as exc:  # noqa: BLE001
            # 调试可视化永远不许把"摔倒靠近"带崩
            rospy.logwarn("发布摔倒调试 Marker 失败（已忽略）：%s", exc)
            self._fall_console("调试 Marker 发布失败（已忽略）：%s" % exc)
            return None

    def _publish_fall_debug_markers(
        self,
        publisher,
        label,
        owner_base,
        samples_base,
        body_base,
        axis_base,
        obb_base,
        standoff_base,
        owner_map,
        goal_map,
        pose_map,
        owner_distance,
        standoff_distance,
        note,
    ):
        markers = []
        has_map = owner_map is not None and pose_map is not None
        frame = self.fall_nav_frame if has_map else self.fall_nav_base_frame

        # 0) 这次"主人位置"到底是用哪些点算出来的
        if samples_base:
            points = [(float(px), float(py), 0.05) for px, py in samples_base]
            markers.append(
                self._fall_debug_points_marker(
                    0, self.fall_nav_base_frame, points, (0.10, 0.85, 0.95, 0.75), 0.035
                )
            )

        # 6) 整条身体带的点（灰）：锚点窗口只有一小撮，想看清"人形轮廓"看这一组
        if body_base:
            points = [(float(px), float(py), 0.03) for px, py in body_base]
            markers.append(
                self._fall_debug_points_marker(
                    6, self.fall_nav_base_frame, points, (0.75, 0.75, 0.75, 0.55), 0.02
                )
            )

        # 7) 身体主方向（黄线）：头↔脚。侧向接近就是沿这条线的垂直方向走的
        if axis_base is not None:
            cx, cy, ax, ay, half = axis_base
            markers.append(
                self._fall_debug_line_marker(
                    7,
                    self.fall_nav_base_frame,
                    [
                        (float(cx) - float(ax) * float(half), float(cy) - float(ay) * float(half), 0.06),
                        (float(cx) + float(ax) * float(half), float(cy) + float(ay) * float(half), 0.06),
                    ],
                    (1.0, 1.0, 0.1),
                    0.04,
                )
            )

        # 8/9) 人体大致形状：地面上的旋转矩形（长边=头↔脚，短边=身体宽）
        if obb_base is not None:
            cx, cy, ux, uy, half_len, half_wid = [float(v) for v in obb_base]
            px, py = -uy, ux
            corners = []
            for sy, sx in ((-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)):
                corners.append(
                    (
                        cx + ux * half_len * sx + px * half_wid * sy,
                        cy + uy * half_len * sx + py * half_wid * sy,
                        0.07,
                    )
                )
            frame = self.fall_nav_base_frame
            # 有 map 位姿就画到 map 系（和机器人、目标点对齐看）。pose_map 必须是
            # (x, y, yaw)；只给了 (x, y) 时退回机器人系，别让 Marker 整块发不出去。
            if has_map and len(pose_map) >= 3:
                frame = self.fall_nav_frame
                x0m, y0m, yaw0m = pose_map[0], pose_map[1], pose_map[2]
                cos_m, sin_m = math.cos(yaw0m), math.sin(yaw0m)
                corners = [
                    (
                        x0m + cos_m * point[0] - sin_m * point[1],
                        y0m + sin_m * point[0] + cos_m * point[1],
                        point[2],
                    )
                    for point in corners
                ]
            markers.append(
                self._fall_debug_line_marker(
                    8, frame, corners, (0.15, 1.0, 0.15, 0.95), 0.05
                )
            )
            if has_map:
                text_x, text_y = corners[0][0], corners[0][1]
            else:
                text_x, text_y = cx, cy
            markers.append(
                self._fall_debug_text_marker(
                    9,
                    frame,
                    text_x,
                    text_y,
                    0.30,
                    "body %.2f x %.2f m" % (2.0 * half_len, 2.0 * half_wid),
                    (0.4, 1.0, 0.4),
                )
            )

        # 1) 主人估计点（红）
        if has_map:
            owner_xy = (owner_map[0], owner_map[1])
        else:
            owner_xy = owner_base
        if owner_xy is not None:
            markers.append(
                self._fall_debug_point_marker(
                    1, frame, owner_xy[0], owner_xy[1], 0.14, (1.0, 0.1, 0.1), 0.14
                )
            )

        # 2) standoff 目标点（绿）
        goal_xy = goal_map if goal_map is not None else standoff_base
        if goal_xy is not None:
            markers.append(
                self._fall_debug_point_marker(
                    2, frame, goal_xy[0], goal_xy[1], 0.10, (0.1, 1.0, 0.2), 0.12
                )
            )

        # 3) 机器人位置（蓝）+ 4/5) 连线
        if has_map:
            markers.append(
                self._fall_debug_point_marker(
                    3, frame, pose_map[0], pose_map[1], 0.05, (0.2, 0.4, 1.0), 0.10
                )
            )
            if owner_xy is not None:
                markers.append(
                    self._fall_debug_line_marker(
                        4,
                        frame,
                        [
                            (pose_map[0], pose_map[1], 0.05),
                            (owner_xy[0], owner_xy[1], 0.05),
                        ],
                        (1.0, 0.6, 0.1),
                    )
                )
            if goal_xy is not None and owner_xy is not None:
                markers.append(
                    self._fall_debug_line_marker(
                        5,
                        frame,
                        [
                            (owner_xy[0], owner_xy[1], 0.05),
                            (goal_xy[0], goal_xy[1], 0.05),
                        ],
                        (0.1, 1.0, 0.2),
                    )
                )
        elif owner_xy is not None:
            markers.append(
                self._fall_debug_line_marker(
                    4,
                    frame,
                    [(0.0, 0.0, 0.05), (owner_xy[0], owner_xy[1], 0.05)],
                    (1.0, 0.6, 0.1),
                )
            )

        # 10/11) 文字标注（ASCII，避免 RViz 里中文乱码）
        if owner_xy is not None:
            if owner_distance is not None:
                text = "owner %.2fm" % float(owner_distance)
            else:
                text = "owner"
            markers.append(
                self._fall_debug_text_marker(
                    10, frame, owner_xy[0], owner_xy[1], 0.45, text, (1.0, 0.4, 0.4)
                )
            )
            if note:
                markers.append(
                    self._fall_debug_text_marker(
                        11, frame, owner_xy[0], owner_xy[1], 0.62, note, (1.0, 1.0, 0.4)
                    )
                )

        # 这次没有数据的 id 显式删掉，免得上一轮的残留误导你
        used_ids = {marker.id for marker in markers}
        for stale_id in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
            if stale_id in used_ids:
                continue
            stale = self._fall_debug_base_marker(stale_id, frame, Marker.SPHERE)
            stale.action = Marker.DELETE
            markers.append(stale)

        publisher.publish(MarkerArray(markers=markers))
        if not self._fall_debug_logged:
            self._fall_debug_logged = True
            rospy.loginfo(
                "摔倒靠近：已发布调试 Marker（%s：红=主人估计点 %.2f m，绿=standoff %.2f m）",
                label,
                owner_distance if owner_distance is not None else -1.0,
                standoff_distance if standoff_distance is not None else -1.0,
            )
        return markers

    def capture_fallen_owner_snapshot(
        self, initial_bbox=None, label="主人", band=None, floor_filter=False
    ):
        """在 ~fall_snapshot_seconds 内多次采样，取中位数作为主人 3D 位置（快照）。"""
        deadline = time.time() + self.fall_snapshot_seconds
        bbox = (
            [float(value) for value in initial_bbox[:4]]
            if initial_bbox is not None and len(initial_bbox) >= 4
            else None
        )
        samples = []
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            image, detections = self.snapshot()
            candidates = (
                self.person_candidates(image, detections) if image is not None else []
            )
            live = self.fall_tracking_candidate(candidates, bbox)
            if live is not None:
                bbox = live.get("bbox", bbox)
            if bbox is not None and image is not None:
                # 优先用"躯干锚点"（左右肩+左右髋中点）采样；关键点拿不到就返回 None，
                # sample_owner_cloud_points 会自动退回整条身体带（行为不变）。
                # ⚠️ 这一段整体包 try：摔倒靠近是"必须做完"的流程，任何采样异常都
                #    不能把动作结果回调/主流程带崩（2026-10-11 现场踩过：出异常后
                #    后续一声不吭，流程直接原地不动）。
                position = None
                try:
                    anchor = self.torso_anchor_from_candidate(live)
                    position = self.owner_position_from_pointcloud(
                        bbox,
                        image.shape,
                        band=band,
                        floor_filter=floor_filter,
                        anchor=anchor,
                    )
                    if position is not None and self.fall_body_axis_enabled:
                        # 再采一次整条身体带：只用来估"头↔脚"方向（决定从哪一侧接近）
                        body_xy = self.body_cloud_points(
                            bbox,
                            image.shape,
                            band=band,
                            floor_filter=floor_filter,
                        )
                        if body_xy:
                            position["body_xy"] = body_xy
                except Exception as exc:  # noqa: BLE001
                    rospy.logwarn("躯干锚点采样失败（%s），退回整条身体带", exc)
                    self._fall_console("躯干锚点采样失败（%s），退回整条身体带" % exc)
                    try:
                        position = self.owner_position_from_pointcloud(
                            bbox, image.shape, band=band, floor_filter=floor_filter
                        )
                    except Exception as exc2:  # noqa: BLE001
                        rospy.logwarn("点云取样失败（%s），本轮跳过这一帧", exc2)
                        self._fall_console("点云取样失败（%s），本轮跳过这一帧" % exc2)
                if position is not None:
                    samples.append(position)
                    if len(samples) >= self.fall_snapshot_min_samples:
                        break
            rate.sleep()
        if not samples:
            return None
        forwards = sorted(sample["forward"] for sample in samples)
        laterals = sorted(sample["lateral"] for sample in samples)
        forward = forwards[len(forwards) // 2]
        lateral = laterals[len(laterals) // 2]
        return {
            "forward": forward,
            "lateral": lateral,
            "distance": math.hypot(forward, lateral),
            "bearing": math.atan2(lateral, max(0.20, forward)),
            "samples": len(samples),
            "points": sum(sample["samples"] for sample in samples),
            "points_xy": samples[-1].get("points_xy", []),
            "body_xy": samples[-1].get("body_xy", []),
            "source": samples[-1].get("source", "band"),
            "bbox": samples[-1].get("bbox"),
        }

    def rotate_toward_bearing(self, bearing, label="主人", timeout=None):
        """原地转到正对 bearing（rad，左正）；返回 True=对准。

        2026-10-16 加**旋转护栏**：这一段是裸发 /cmd_vel、只看里程计，车体旋转时
        会把侧面/斜后方的东西（椅子腿、桌腿）横扫进去。所以每拍都查一次
        `scan_any_min_distance()`（全向），贴到车体（≤ ~scan_turn_guard_distance）
        就立刻停止转动并返回 False —— 宁可没转正，也不许撞。
        """
        limit = (
            self.fall_snapshot_align_timeout if timeout is None else max(0.5, float(timeout))
        )
        start_yaw = self.get_latest_yaw()
        if start_yaw is None:
            rospy.logwarn("%s：没有里程计 yaw，跳过转正直接直行", label)
            return False
        target_yaw = self.wrap_angle(start_yaw + float(bearing))
        deadline = time.time() + limit
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            guard = getattr(self, "scan_any_min_distance", None)
            turn_guard = None
            if guard is not None:
                try:
                    turn_guard = guard()
                except Exception:  # noqa: BLE001 - 护栏坏了也不能把靠近流程带崩
                    turn_guard = None
            guard_limit = float(getattr(self, "scan_turn_guard_distance", 0.0) or 0.0)
            if turn_guard is not None and guard_limit > 0.0 and float(turn_guard) <= guard_limit:
                self.stop_base()
                rospy.logwarn(
                    "%s：旋转护栏触发（全向最近 %.2f m ≤ %.2f m），停止转正"
                    "（旁边有东西贴着车体，转过去就会撞）",
                    label,
                    float(turn_guard),
                    guard_limit,
                )
                try:
                    print(
                        "[旋转护栏] %s：全向最近 %.2f m ≤ %.2f m，停止转正"
                        % (label, float(turn_guard), guard_limit),
                        flush=True,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return False
            yaw = self.get_latest_yaw()
            if yaw is None:
                rate.sleep()
                continue
            error = self.wrap_angle(target_yaw - yaw)
            if abs(error) <= self.fall_snapshot_bearing_tolerance:
                self.stop_base()
                rospy.loginfo(
                    "%s：已按 bearing %.3f rad 转正（残差 %.3f rad）", label, bearing, error
                )
                return True
            twist = Twist()
            twist.angular.z = max(
                -self.fall_snapshot_turn_speed,
                min(self.fall_snapshot_turn_speed, 1.5 * error),
            )
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.stop_base()
        rospy.logwarn("%s：转正超时（bearing %.3f rad 未到位）", label, bearing)
        return False

    def drive_forward_until_cloud_close(self, bbox, label="主人", stop_distance=None):
        """直行，直到点云说"人已经近到快看不见"（或护栏/超时）。

        返回 (last_distance, reason)：
          cloud_stop  = 云测到 ≤ stop_distance（默认 fall_cloud_stop_distance）
          cloud_lost  = 连续 0.5 s 量不到人（多半是掉出画面了）→ 该切爬行了
          scan_guard  = 前方障碍太近
          timeout     = 直行超时
        """
        effective_stop = (
            self.fall_cloud_stop_distance
            if stop_distance is None
            else max(0.35, float(stop_distance))
        )
        deadline = time.time() + self.fall_snapshot_drive_timeout
        last_distance = None
        cloud_lost_since = None
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            image, detections = self.snapshot()
            candidates = (
                self.person_candidates(image, detections) if image is not None else []
            )
            position = None
            if candidates:
                live = None
                if bbox is not None:
                    best = max(
                        candidates,
                        key=lambda item: self.bbox_iou(bbox, item.get("bbox")),
                    )
                    if self.bbox_iou(bbox, best.get("bbox")) > 0.05:
                        live = best
                if live is None and bbox is None:
                    live = candidates[0]
                if live is not None:
                    bbox = live.get("bbox", bbox)
                    if image is not None:
                        position = self.owner_position_from_pointcloud(bbox, image.shape)

            twist = Twist()
            twist.linear.x = self.fall_snapshot_speed
            if position is not None:
                last_distance = position["distance"]
                cloud_lost_since = None
                # 直行段只做"小幅纠偏"，并且带死区：侧向偏差已经很小就不动方向，
                # 否则点云侧向量的固定偏差会让机器人一路"边直行边微转"画弧。
                bearing = float(position["bearing"])
                if abs(bearing) <= self.fall_snapshot_bearing_tolerance:
                    twist.angular.z = 0.0
                else:
                    twist.angular.z = max(-0.35, min(0.35, 0.9 * bearing))
                if last_distance <= effective_stop:
                    self.stop_base()
                    rospy.loginfo(
                        "%s：点云距离 %.2f m ≤ %.2f m，停止直行，准备爬行收尾",
                        label,
                        last_distance,
                        effective_stop,
                    )
                    return last_distance, "cloud_stop"
            else:
                twist.angular.z = 0.0
                if cloud_lost_since is None:
                    cloud_lost_since = time.time()
                elif time.time() - cloud_lost_since >= 0.5:
                    self.stop_base()
                    rospy.loginfo(
                        "%s：连续 0.5 s 量不到人（多半已近到掉出画面），停止直行",
                        label,
                    )
                    return last_distance, "cloud_lost"

            guard = self.scan_forward_min_distance()
            if guard is not None and guard <= self.fall_scan_guard_distance:
                self.stop_base()
                rospy.logwarn(
                    "%s：直行中止，前方 %.2f m 有障碍（护栏 %.2f m）",
                    label,
                    guard,
                    self.fall_scan_guard_distance,
                )
                return last_distance, "scan_guard"

            self.cmd_pub.publish(twist)
            rate.sleep()
        self.stop_base()
        return last_distance, "timeout"

    def creep_forward_by_odometry(self, distance, label="主人", scan_stop_distance=None):
        """按里程计直行 distance 米（慢速 + /scan 护栏）；返回 (ok, travelled)。

        scan_stop_distance 不为空时，还会把雷达正前方距离当"贴近了没有"的判据：
        走到 ≤ 该值就停（躺在地上的人在近处相机看不到，但雷达一直看得到）。
        """
        target = max(0.0, float(distance))
        if target <= 0.0:
            return True, 0.0
        x0, y0, age = self.get_latest_odom_xy()
        if x0 is None or age is None or age > 1.0:
            # 没有里程计位置：退化成定时前进（比不动强，日志里写明）
            seconds = target / max(0.03, self.fall_creep_speed)
            rospy.logwarn(
                "%s：没有可用的里程计位置，爬行段改为定时前进 %.1f s（约 %.2f m）",
                label,
                seconds,
                target,
            )
            deadline = time.time() + min(seconds, self.fall_creep_timeout)
            rate = rospy.Rate(10)
            while not rospy.is_shutdown() and time.time() < deadline:
                guard = self.scan_forward_min_distance()
                if guard is not None and guard <= self.fall_scan_guard_distance:
                    self.stop_base()
                    rospy.logwarn("%s：定时爬行中止，前方 %.2f m 有障碍", label, guard)
                    return False, 0.0
                if scan_stop_distance is not None and guard is not None and guard <= scan_stop_distance:
                    self.stop_base()
                    rospy.loginfo(
                        "%s：雷达前方 %.2f m ≤ %.2f m，判定已贴到主人，停",
                        label,
                        guard,
                        scan_stop_distance,
                    )
                    return True, 0.0
                twist = Twist()
                twist.linear.x = self.fall_creep_speed
                self.cmd_pub.publish(twist)
                rate.sleep()
            self.stop_base()
            return True, target

        deadline = time.time() + self.fall_creep_timeout
        travelled = 0.0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            x, y, _age = self.get_latest_odom_xy()
            if x is None:
                break
            travelled = math.hypot(x - x0, y - y0)
            if target - travelled <= self.fall_creep_finish_tolerance:
                self.stop_base()
                return True, travelled
            guard = self.scan_forward_min_distance()
            if guard is not None and guard <= self.fall_scan_guard_distance:
                self.stop_base()
                rospy.logwarn(
                    "%s：爬行中止，前方 %.2f m 有障碍（护栏 %.2f m），已走 %.2f m / %.2f m",
                    label,
                    guard,
                    self.fall_scan_guard_distance,
                    travelled,
                    target,
                )
                return False, travelled
            if scan_stop_distance is not None and guard is not None and guard <= scan_stop_distance:
                self.stop_base()
                rospy.loginfo(
                    "%s：雷达前方 %.2f m ≤ %.2f m，判定已贴到主人，停（已走 %.2f m）",
                    label,
                    guard,
                    scan_stop_distance,
                    travelled,
                )
                return True, travelled
            twist = Twist()
            twist.linear.x = self.fall_creep_speed
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.stop_base()
        return travelled + self.fall_creep_finish_tolerance >= target, travelled

    def approach_fallen_owner_snapshot(self, owner_name="", initial_bbox=None):
        """摔倒靠近（快照式）：快照 3D 位置 → 转正 → 直行到云快看不见 → 里程计爬完。

        返回 True=按计划走完（或走完能走的部分）；False=刚开始就拿不到位置（调用方
        会退回原来的"看着人闭环"）。
        """
        target = float(self.fall_approach_standoff_distance)
        label = "%s（摔倒·快照靠近）" % (owner_name or "主人")
        if not self.ensure_approach_pointcloud():
            rospy.logwarn("%s：点云不可用，退回相机闭环", label)
            return False
        self.ensure_scan_subscription()
        # 与 B-2 用同一套取样：整条身体带 + 地面剔除（避免只取到身体某一端）
        band = (
            float(getattr(self, "fall_owner_band_low", 0.15)),
            float(getattr(self, "fall_owner_band_high", 0.85)),
        )
        snapshot = self.capture_fallen_owner_snapshot(
            initial_bbox, label=label, band=band, floor_filter=True
        )
        if snapshot is None:
            # 兜底：用 B-2 在"摔倒确认瞬间"冻结的那份位置（用户要求：兜底也要能贴到 0.5 m，
            # 不能因为"现在看不见人"就放弃）。方位角按当前姿态近似为 0（人就在正前方）。
            frozen = getattr(self, "_fall_frozen_snapshot", None)
            if frozen:
                rospy.logwarn(
                    "%s：现在拿不到新的点云位置，改用摔倒瞬间冻结的位置"
                    "（前=%.2f 侧=%.2f）继续",
                    label,
                    float(frozen.get("forward") or 0.0),
                    float(frozen.get("lateral") or 0.0),
                )
                self._fall_console("%s：改用冻结位置继续（兜底）" % label)
                snapshot = dict(frozen)
                snapshot["bearing"] = 0.0
            else:
                rospy.logwarn("%s：快照失败（拿不到主人点云位置），退回相机闭环", label)
                return False
        rospy.loginfo(
            "%s：快照位置 前=%.2f m 侧=%.2f m 距离=%.2f m bearing=%.3f rad"
            "（样本 %d 次 / %d 点）→ 先转正",
            label,
            snapshot["forward"],
            snapshot["lateral"],
            snapshot["distance"],
            snapshot["bearing"],
            snapshot["samples"],
            snapshot["points"],
        )
        if str(self.approach_start_text or "").strip():
            # 只在第一次进"摔倒靠近"时播一次（B-2 失败回退到 B-1 不再重复播报）
            self.fall_speak_approach_start_once()

        # ① 转正（按点云算出的 bearing，不看画面框中心）。
        #    只有"明显偏"才转：几度的偏差多半来自身体的横向延伸而不是"人在侧面"，
        #    上一轮现场就是被这 9° 的小转正带偏成"朝头/脚开"的。
        if abs(snapshot["bearing"]) > max(self.fall_snapshot_bearing_tolerance, 0.15):
            self.rotate_toward_bearing(snapshot["bearing"], label=label)
        else:
            rospy.loginfo(
                "%s：bearing %.3f rad 很小，不转正，直接按当前朝向直行", label, snapshot["bearing"]
            )

        # ② 直行到"人快掉出画面"（或云已 ≤ 切换距离）
        last_distance, reason = self.drive_forward_until_cloud_close(
            snapshot.get("bbox"), label=label
        )

        # ③ 剩下的距离用里程计爬完（相机在这个距离已经看不到躺着的人）
        if last_distance is not None:
            creep = max(0.0, min(self.fall_creep_max_distance, last_distance - target))
        else:
            creep = min(
                self.fall_creep_max_distance,
                max(0.0, snapshot["distance"] - self.fall_cloud_stop_distance),
            )
        rospy.loginfo(
            "%s：直行结束（%s，最后云距离=%s）→ 里程计爬行 %.2f m（目标 %.2f m，"
            "护栏 %.2f m）",
            label,
            reason,
            "无" if last_distance is None else "%.2f m" % last_distance,
            creep,
            target,
            self.fall_scan_guard_distance,
        )
        ok, travelled = self.creep_forward_by_odometry(creep, label=label)
        rospy.loginfo(
            "%s：结束 —— 爬行 %.2f m，成功=%s（%s）", label, travelled, ok, reason
        )
        return True

    # ------------------------------------------------------------------
    # 摔倒靠近：B-2（快照 + move_base 避障 + 短距补齐）
    # ------------------------------------------------------------------
    @staticmethod
    def _fall_console(message):
        """摔倒靠近的"终端可见"日志（print 到 stdout）。

        为什么需要它：动作识别一触发，任务节点的 /rosout 与日志文件就会失灵
        （10-11 20:18 那几次跑就有，跟本次改动无关），摔倒靠近整段成了"黑屏"。
        print 走 stdout（VS Code 集成终端 + PYTHONUNBUFFERED=1）一定看得见，
        所以关键步骤都补一条，方便现场判断走到哪一步。
        """
        try:
            print("[摔倒靠近] %s" % message, flush=True)
        except Exception:  # noqa: BLE001
            pass

    def fall_speak_approach_start_once(self):
        """摔倒靠近开始时那句"我这就过来"**只说一次**。

        现场问题（2026-10-12）：B-2 失败回退到 B-1 时，两个分支各播一遍，
        听起来像"连续播报我正在过去"。用一个标志位兜住，整次靠近只播一次。
        """
        if getattr(self, "_fall_approach_announced", False):
            return False
        self._fall_approach_announced = True
        if str(self.approach_start_text or "").strip():
            self.speak(self.approach_start_text, wait=False)
            return True
        return False

    def robot_pose_in_map(self, timeout=1.0):
        """机器人当前位姿（map 系）→ (x, y, yaw, age)；拿不到返回 (None, None, None, None)。

        timeout 可调小：看护循环里每 0.5 秒查一次，用 1 秒的等待会把循环拖慢。
        """
        if self._tf_buffer is None:
            try:
                import tf2_ros

                self._tf_buffer = tf2_ros.Buffer()
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
                rospy.loginfo(
                    "摔倒靠近：已建立 TF 监听（%s→%s）用来把 standoff 点换算到 map 系",
                    self.fall_nav_frame,
                    self.fall_nav_base_frame,
                )
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn("tf2_ros 不可用（无法把目标换算到 map 系）：%s", exc)
                self._tf_buffer = False
        if self._tf_buffer is False:
            return None, None, None, None
        try:
            transform = self._tf_buffer.lookup_transform(
                self.fall_nav_frame,
                self.fall_nav_base_frame,
                rospy.Time(0),
                rospy.Duration(max(0.05, float(timeout))),
            )
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                5.0,
                "查 TF %s→%s 失败：%s",
                self.fall_nav_frame,
                self.fall_nav_base_frame,
                exc,
            )
            return None, None, None, None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        stamp = transform.header.stamp.to_sec()
        age = max(0.0, rospy.Time.now().to_sec() - stamp) if stamp > 0.0 else 0.0
        return float(translation.x), float(translation.y), yaw, age

    def fall_nav_plan_ok(self, start_xy, goal_xy):
        """用 /move_base/make_plan 预检"这条路径是不是太绕"。返回 (ok, reason)。

        参考实现（server_robot3）用同样的手段挡掉"绕着半个房间去摸人"的规划。
        拿不到服务时不挡路（只记日志）——预检是加分项，不是门槛。
        """
        if not self.fall_nav_plan_check:
            return True, "预检关闭"
        if self._make_plan is None:
            try:
                from nav_msgs.srv import GetPlan

                self._make_plan = rospy.ServiceProxy(
                    self.fall_nav_plan_service, GetPlan
                )
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn("拿不到 %s（跳过路径预检）：%s", self.fall_nav_plan_service, exc)
                self._make_plan = False
        if self._make_plan is False:
            return True, "make_plan 不可用，跳过预检"
        try:
            from geometry_msgs.msg import PoseStamped

            start = PoseStamped()
            start.header.frame_id = self.fall_nav_frame
            start.header.stamp = rospy.Time.now()
            start.pose.position.x = float(start_xy[0])
            start.pose.position.y = float(start_xy[1])
            start.pose.orientation.w = 1.0
            goal = PoseStamped()
            goal.header.frame_id = self.fall_nav_frame
            goal.header.stamp = start.header.stamp
            goal.pose.position.x = float(goal_xy[0])
            goal.pose.position.y = float(goal_xy[1])
            goal.pose.orientation.w = 1.0
            response = self._make_plan(start=start, goal=goal, tolerance=0.15)
            poses = list(getattr(response.plan, "poses", []) or [])
            if len(poses) < 2:
                return False, "make_plan 返回空路径"
            length = 0.0
            for index in range(1, len(poses)):
                dx = poses[index].pose.position.x - poses[index - 1].pose.position.x
                dy = poses[index].pose.position.y - poses[index - 1].pose.position.y
                length += math.hypot(dx, dy)
            straight = math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])
            ratio = length / max(0.05, straight)
            if ratio > self.fall_nav_plan_detour_ratio:
                return False, "路径太绕：路径 %.2f m / 直线 %.2f m = %.2f（上限 %.2f）" % (
                    length,
                    straight,
                    ratio,
                    self.fall_nav_plan_detour_ratio,
                )
            return True, "路径 %.2f m / 直线 %.2f m = %.2f" % (length, straight, ratio)
        except Exception as exc:  # noqa: BLE001
            return True, "make_plan 调用失败，跳过预检：%s" % exc

    def guard_obstacle_is_owner(self, owner_bbox):
        """护栏触发时判断"挡在前面的到底是不是主人"。

        低装雷达（0.1 m 高）看得到**躺在地上的人**，所以靠近到身体前会报 0.4 m 障碍
        —— 但现场也踩过"主人头旁边 20 cm 处放着一个纸箱"，那时报的是箱子，不是人。
        所以判据不能只看"有没有障碍"，要**对得上位置**：

          ① 这一点云帧量不到主人 → 人已经太近/出画面 → 认定是主人；
          ② 量得到主人，但主人还远（> ~fall_nav_guard_arrival_distance）→ 不是主人；
          ③ 量得到主人且很近，但雷达障碍比主人**近太多**
             （差值 > ~fall_nav_guard_match_tolerance，默认 0.6 m）→ 前面挡着的是别的东西。
        """
        try:
            image, _ = self.snapshot()
            if image is None or owner_bbox is None or len(owner_bbox) < 4:
                return False
            distance = self.owner_distance_from_pointcloud(owner_bbox, image.shape)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn("护栏判人失败（按\"不是主人\"处理）：%s", exc)
            return False
        if distance is None:
            return True
        distance = float(distance)
        if distance > self.fall_nav_guard_arrival_distance:
            return False
        guard = self.scan_forward_min_distance()
        if guard is not None and (distance - float(guard)) > self.fall_nav_guard_match_tolerance:
            return False
        return True

    def navigate_fall_goal_with_guard(self, goal, timeout, label, owner_bbox=None):
        """发 move_base 目标并等结果；期间同时盯**雷达护栏**和**看护循环**。

        2026-10-16 加的看护循环（移植自 server_robot3 的
        send_approach_navigation_goal），治的是实测踩过的两个坑：
          · move_base 报"成功"时其实还差一截 —— 用里程计估算剩余，差太多判失败；
          · 规划器/局部规划器卡住不动 —— 不用干等满 25 秒，提前取消并说明原因。
        另外有"慢停接收"：剩余已经很小、而且实际也不动了，就当作到位
        （局部规划器在终点前反复微调时，靠它把这一段收掉）。

        雷达硬护栏保持最高优先级：撞到之前先停。返回 (ok, reason)。
        """
        if not self.move_base.wait_for_server(
            rospy.Duration(min(5.0, self.move_base_server_timeout))
        ):
            return False, "move_base action server 不可用"

        # ---- 发目标前的准备：这一段要走多远、起点里程计在哪 ----
        # 剩余距离用"这一段起点到目标的直线距离 − 里程计已走的弧长"来估。
        # 为什么不每拍都查 TF：robot_pose_in_map 在 TF 不通时会阻塞，
        # 10 Hz 的循环会被拖死；里程计是本地数据，够用且不会卡。
        goal_xy = (
            float(goal.target_pose.pose.position.x),
            float(goal.target_pose.pose.position.y),
        )
        leg_distance = None
        if (
            self.fall_nav_watchdog_enabled
            and str(goal.target_pose.header.frame_id or "") == self.fall_nav_frame
        ):
            pose_x, pose_y, _pose_yaw, _pose_age = self.robot_pose_in_map(timeout=0.3)
            if pose_x is not None:
                leg_distance = math.hypot(goal_xy[0] - pose_x, goal_xy[1] - pose_y)
        start_xy = None
        odom_x, odom_y, odom_age = self.get_latest_odom_xy()
        if odom_x is not None and (odom_age is None or odom_age <= 1.0):
            start_xy = (odom_x, odom_y)

        # 这条日志两条路共用（摔倒靠近 / 挥手靠近），所以别写死"摔倒"字样
        self.clear_move_base_costmaps("靠近前（%s）" % label)
        send_time = time.time()
        self.move_base.send_goal(goal)
        deadline = send_time + max(2.0, float(timeout))
        best_remaining = leg_distance
        last_progress_time = send_time
        slow_finish_cycles = 0
        remaining = None
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            now = time.time()
            # ---- ① 雷达硬护栏（最高优先级，撞之前先停）----
            guard = self.scan_forward_min_distance()
            if guard is not None and guard <= self.fall_nav_guard_distance:
                self.move_base.cancel_goal()
                self.stop_base()
                # 拦在前面的很可能就是躺在地上的主人（雷达 0.1 m 高，看得到人）。
                # 这种情况不要"放弃导航退回下一级"（实测一退整套动作就乱），
                # 而是当成"已经走到跟前"，直接进补齐/协助。
                if self.fall_nav_guard_is_arrival and self.guard_obstacle_is_owner(
                    owner_bbox
                ):
                    self._fall_console(
                        "%s：护栏触发 %.2f m，且主人就在跟前 → 当作到位（进补齐/协助）"
                        % (label, guard)
                    )
                    return True, "前向护栏触发（前方 %.2f m），主人就在跟前 → 当作到位" % guard
                self._fall_console(
                    "%s：护栏触发 %.2f m，但对不上主人位置 → 判定是真障碍，取消导航"
                    % (label, guard)
                )
                return False, "前方 %.2f m 有障碍（护栏 %.2f m），取消导航" % (
                    guard,
                    self.fall_nav_guard_distance,
                )

            # ---- ② 看护循环：剩余距离 / 卡死 / 慢停 ----
            remaining = None
            travelled = None
            linear_speed = None
            angular_speed = None
            if self.fall_nav_watchdog_enabled:
                current_x, current_y, current_age = self.get_latest_odom_xy()
                if (
                    start_xy is not None
                    and current_x is not None
                    and (current_age is None or current_age <= 1.0)
                ):
                    travelled = math.hypot(
                        current_x - start_xy[0], current_y - start_xy[1]
                    )
                if leg_distance is not None and travelled is not None:
                    remaining = max(0.0, leg_distance - travelled)
                    if remaining + self.fall_nav_stuck_min_progress < (
                        best_remaining if best_remaining is not None else remaining + 1.0
                    ):
                        best_remaining = remaining
                        last_progress_time = now
                linear_speed, angular_speed, _speed_age = self.get_latest_odom_speeds(
                    max_age=self.fall_nav_odom_speed_max_age
                )
                # 还在走（或还在原地转正）都算"有进展"，不算卡死。
                if linear_speed is not None and linear_speed > self.fall_nav_stuck_speed:
                    last_progress_time = now
                if (
                    angular_speed is not None
                    and abs(angular_speed) > self.fall_nav_stuck_angular_speed
                ):
                    last_progress_time = now

                if (
                    self.fall_nav_slow_finish_enabled
                    and remaining is not None
                    and linear_speed is not None
                    and remaining <= self.fall_nav_slow_finish_tolerance
                    and linear_speed <= self.fall_nav_slow_finish_speed
                ):
                    slow_finish_cycles += 1
                    if slow_finish_cycles >= self.fall_nav_slow_finish_cycles:
                        self.move_base.cancel_goal()
                        self.stop_base()
                        self._fall_console(
                            "%s：慢停接收（剩余 %.2f m，实测速度 %.3f m/s）→ 当作到位"
                            % (label, remaining, linear_speed)
                        )
                        return True, "慢停接收（剩余 %.2f m）" % remaining
                else:
                    slow_finish_cycles = 0

                if (
                    remaining is not None
                    and now - last_progress_time >= self.fall_nav_stuck_timeout
                    and linear_speed is not None
                    and linear_speed <= self.fall_nav_stuck_speed
                    and (
                        angular_speed is None
                        or abs(angular_speed) <= self.fall_nav_stuck_angular_speed
                    )
                ):
                    self.move_base.cancel_goal()
                    self.stop_base()
                    return False, (
                        "卡住/被挡住：%.0f 秒没有进展（剩余 %.2f m，速度 %.3f m/s）"
                        % (self.fall_nav_stuck_timeout, remaining, linear_speed)
                    )

                if remaining is not None:
                    rospy.loginfo_throttle(
                        1.0,
                        "%s：导航中 已走 %.2f m / 剩余约 %.2f m，速度 %.3f m/s，护栏 %s",
                        label,
                        travelled if travelled is not None else -1.0,
                        remaining,
                        linear_speed if linear_speed is not None else -1.0,
                        "无" if guard is None else "%.2f m" % guard,
                    )

            # ---- ③ move_base 自己的状态 ----
            state = self.move_base.get_state()
            # 刚发出去的头 0.5 秒状态可能是上一单的残留，先不理会。
            if now - send_time >= 0.5:
                if state == GoalStatus.SUCCEEDED:
                    self.stop_base()
                    if remaining is None or remaining <= self.fall_nav_arrival_tolerance:
                        return True, ""
                    return False, (
                        "move_base 报到达，但里程计估算还差 %.2f m（容差 %.2f）"
                        % (remaining, self.fall_nav_arrival_tolerance)
                    )
                if state in (
                    GoalStatus.PREEMPTED,
                    GoalStatus.ABORTED,
                    GoalStatus.REJECTED,
                    GoalStatus.RECALLED,
                    GoalStatus.LOST,
                ):
                    self.stop_base()
                    return False, "move_base 状态 %s（未到达）" % state

            self.update_yolo_window(label)
            rate.sleep()

        self.move_base.cancel_goal()
        self.stop_base()
        if rospy.is_shutdown():
            return False, "ROS 已关闭"
        return False, "导航超时（%.0f 秒）" % timeout

    def fall_finish_approach(self, bbox, label, stage_distance=None):
        """最后一段：**直行闭环**到 fall_direct_stop_distance → **盲走**补齐（两段都带 /scan 护栏）。

        2026-10-12 与用户商定：
          · 直行段用点云量距离（走到 ≈0.75 m）；
          · 之后盲走 0.25 m 到 ≈0.5 m —— 1 m 以内相机看不到躺着的人，只能靠里程计，
            但雷达软停车线（fall_extra_close_scan_stop_distance 0.55 m）始终开着，
            撞到人或家具会先停。
        返回 True（这一步不判定成败，到了就交给后面的协助动作）。
        """
        target = float(self.fall_approach_standoff_distance)
        stop = float(self.fall_direct_stop_distance)
        measured, reason = self.drive_forward_until_cloud_close(
            bbox, label=label, stop_distance=stop
        )
        self._fall_console(
            "%s：直行段结束（%s，点云最后距离=%s，停线 %.2f m）"
            % (
                label,
                reason,
                "无" if measured is None else "%.2f m" % float(measured),
                stop,
            )
        )
        if measured is not None:
            remaining = float(measured) - target
        else:
            # 直行段点云先丢了（多半已经到 1 m 以内）：按"从停线再补一点"估
            remaining = max(0.0, stop - target) + 0.15
        remaining = max(0.0, min(self.fall_extra_close_max_distance, remaining))
        if remaining > self.fall_creep_finish_tolerance:
            done, travelled = self.creep_forward_by_odometry(
                remaining,
                label=label,
                scan_stop_distance=self.fall_extra_close_scan_stop_distance,
            )
            rospy.loginfo(
                "%s：盲走补齐 %.2f m / %.2f m 完成=%s（雷达软停车线 %.2f m）",
                label,
                travelled,
                remaining,
                done,
                self.fall_extra_close_scan_stop_distance,
            )
            self._fall_console(
                "%s：盲走 %.2f m / %.2f m 完成=%s" % (label, travelled, remaining, done)
            )
        else:
            rospy.loginfo("%s：直行后已在目标距离内（还差 %.2f m）", label, remaining)
        return True

    def approach_fallen_owner_navigate(self, owner_name="", initial_bbox=None):
        """摔倒靠近（B-2）：快照 3D 位置 → map 系 standoff 目标交 move_base → 里程计补齐。

        返回 True=导航段走通（含最后补齐）；False=没能开始/没走通（调用方继续退回
        B-1 快照式、再退回相机闭环）。
        """
        target = float(self.fall_approach_standoff_distance)
        label = "%s（摔倒·导航靠近）" % (owner_name or "主人")
        self._fall_console("%s：进入 B-2（目标 %.2f m）" % (label, target))
        if not self.ensure_approach_pointcloud():
            rospy.logwarn("%s：点云不可用，退回下一级", label)
            self._fall_console("%s：点云不可用 → 退回下一级" % label)
            return False
        self.ensure_scan_subscription()
        band = (self.fall_owner_band_low, self.fall_owner_band_high)
        snapshot = self.capture_fallen_owner_snapshot(
            initial_bbox, label=label, band=band, floor_filter=True
        )
        if snapshot is None:
            rospy.logwarn("%s：快照失败（拿不到主人点云位置），退回下一级", label)
            self._fall_console("%s：快照失败（拿不到主人点云位置）→ 退回下一级" % label)
            return False
        distance = float(snapshot["distance"])
        self._fall_console(
            "%s：快照 采样=%s 前=%.2f 侧=%.2f 距离=%.2f bearing=%.3f（%d 点）"
            % (
                label,
                "躯干锚点" if snapshot.get("source") == "anchor" else "整条身体带",
                float(snapshot["forward"]),
                float(snapshot["lateral"]),
                distance,
                float(snapshot["bearing"]),
                int(snapshot.get("points") or 0),
            )
        )
        rospy.loginfo(
            "%s：快照位置 前=%.2f m 侧=%.2f m 距离=%.2f m bearing=%.3f rad"
            "（采样=%s（带 %.0f%%~%.0f%%）+ 地面剔除；样本 %d 次 / %d 点）",
            label,
            snapshot["forward"],
            snapshot["lateral"],
            distance,
            snapshot["bearing"],
            "躯干锚点" if snapshot.get("source") == "anchor" else "整条身体带",
            self.fall_owner_band_low * 100.0,
            self.fall_owner_band_high * 100.0,
            snapshot["samples"],
            snapshot["points"],
        )
        # 摔倒靠近开始那句"我这就过来"：整次靠近只播一次（B-2/B-1/相机闭环共用同一个标志）
        self.fall_speak_approach_start_once()

        # ---- 阶段 1：先原地转正对准躯干（用户要求：先正对，再靠近）------------
        # 为什么要有这一步：move_base 的目标朝向是"从目标点看向躯干"，如果当前朝向
        # 差得多（实测差过 50°），局部规划器一上来就原地大转弯，现场看起来就是
        # "往一边跑、完全不是人的方向"。先原地转正，后面就是一条近乎直线的靠近。
        bearing = float(snapshot.get("bearing") or 0.0)
        if self.fall_face_first_enabled and abs(bearing) > self.fall_face_first_max_angle:
            self._fall_console(
                "%s：阶段1 先转正对准躯干（bearing=%.3f rad）" % (label, bearing)
            )
            aligned = self.rotate_toward_bearing(bearing, label=label)
            self._fall_console(
                "%s：阶段1 %s" % (label, "已对准" if aligned else "转正超时（继续按原位置算）")
            )
            if aligned:
                again = self.capture_fallen_owner_snapshot(
                    snapshot.get("bbox"), label=label, band=band, floor_filter=True
                )
                if again is not None:
                    snapshot = again
                    distance = float(snapshot["distance"])
                    self._fall_console(
                        "%s：转正后复测 前=%.2f 侧=%.2f 距离=%.2f bearing=%.3f"
                        % (
                            label,
                            float(snapshot["forward"]),
                            float(snapshot["lateral"]),
                            distance,
                            float(snapshot.get("bearing") or 0.0),
                        )
                    )

        # 主人在机器人系里的位置：点云量的是"到相机"，补上相机相对 base 的前向偏移。
        # 这一段与位姿无关，先算出来：① "已在 standoff 附近"分支要用；
        # ② 调试可视化要用；③ 后面转 map 系只是加一次旋转平移。
        owner_x = float(snapshot["forward"]) + self.approach_camera_forward_offset
        owner_y = float(snapshot["lateral"])
        # 冻结"摔倒瞬间认定的主人位置"：B-2 万一半路失败，兜底也用它（不再靠相机重新定位）
        self._fall_frozen_snapshot = dict(snapshot)
        # 中间点用 stage（默认 1.0 m，距离不必精确）；最后一段再直行到 0.75 m + 盲走 0.25 m
        standoff = max(0.60, float(self.fall_nav_stage_standoff_distance))
        # 目标点：默认在"主人→机器人"连线上退 standoff；有身体主方向时退到**身体侧面**
        # （避免从头/脚方向怼过去），并保证目标点离任何身体点 ≥ fall_nav_body_clearance。
        goal_x_base, goal_y_base, face_yaw_rel, goal_info = self.fall_standoff_goal_base(
            owner_x,
            owner_y,
            standoff,
            body_xy=snapshot.get("body_xy"),
            label=label,
        )
        self._fall_console(
            "%s：接近方式=%s 有效standoff=%.2f m 离身体最近=%.2f m（身体半长=%s）"
            % (
                label,
                "身体侧面" if goal_info.get("mode") == "side" else "沿视线",
                float(goal_info.get("standoff") or standoff),
                float(goal_info.get("min_body_distance") or -1.0),
                "n/a"
                if goal_info.get("spread") is None
                else "%.2f" % float(goal_info["spread"]),
            )
        )
        axis_marker = None
        axis_vector = goal_info.get("axis")
        if axis_vector is not None:
            half = max(0.30, min(0.90, float(goal_info.get("spread") or 0.45)))
            axis_marker = (
                owner_x,
                owner_y,
                float(axis_vector[0]),
                float(axis_vector[1]),
                half,
            )

        # 已经在中间点附近：不导航，直接进"直行闭环 + 盲走收尾"
        if distance - standoff <= self.fall_nav_distance_tolerance:
            rospy.loginfo(
                "%s：已在中间点附近（%.2f m ≤ %.2f + %.2f），跳过导航直接直行+盲走",
                label,
                distance,
                standoff,
                self.fall_nav_distance_tolerance,
            )
            # 这个分支还没查 TF，Marker 就按机器人系画（RViz 会用 TF 摆正）
            self.publish_fall_debug_markers(
                label,
                owner_base=(owner_x, owner_y),
                samples_base=snapshot.get("points_xy"),
                body_base=snapshot.get("body_xy"),
                axis_base=axis_marker,
                obb_base=goal_info.get("obb"),
                standoff_base=(goal_x_base, goal_y_base),
                owner_distance=distance,
                standoff_distance=standoff,
                note="already near standoff",
            )
            self._fall_console(
                "%s：已在中间点附近（%.2f m），跳过导航，直接直行+盲走收尾"
                % (label, distance)
            )
            return self.fall_finish_approach(snapshot.get("bbox"), label)

        pose = self.robot_pose_in_map()
        if pose[0] is None:
            rospy.logwarn("%s：拿不到 map 系位姿（TF 不通），退回下一级", label)
            self._fall_console("%s：拿不到 map 位姿（TF 不通）→ 退回下一级" % label)
            return False
        x0, y0, yaw0, age = pose
        goal_x = x0 + math.cos(yaw0) * goal_x_base - math.sin(yaw0) * goal_y_base
        goal_y = y0 + math.sin(yaw0) * goal_x_base + math.cos(yaw0) * goal_y_base
        # 朝向：从目标点看向身体中心（侧向接近时就是"正对躯干"）
        goal_yaw = self.wrap_angle(yaw0 + float(face_yaw_rel))
        owner_map_x = x0 + math.cos(yaw0) * owner_x - math.sin(yaw0) * owner_y
        owner_map_y = y0 + math.sin(yaw0) * owner_x + math.cos(yaw0) * owner_y
        rospy.loginfo(
            "%s：主人 map≈(%.2f, %.2f)，目标（%s standoff %.2f m，离身体最近 %.2f m）"
            "map=(%.2f, %.2f) yaw=%.2f；"
            "机器人 map=(%.2f, %.2f) yaw=%.2f（位姿 age=%.1fs）；travel 上限 %.2f m",
            label,
            owner_map_x,
            owner_map_y,
            "身体侧面" if goal_info.get("mode") == "side" else "沿视线",
            float(goal_info.get("standoff") or standoff),
            float(goal_info.get("min_body_distance") or -1.0),
            goal_x,
            goal_y,
            goal_yaw,
            x0,
            y0,
            yaw0,
            age if age is not None else -1.0,
            self.fall_nav_max_travel,
        )
        # 调试可视化（不影响任何决策）：红=主人估计点，绿=standoff 目标点，
        # 蓝=机器人，青=这次用到的采样点，橙线=机器人→主人。
        self.publish_fall_debug_markers(
            label,
            owner_base=(owner_x, owner_y),
            samples_base=snapshot.get("points_xy"),
            body_base=snapshot.get("body_xy"),
            axis_base=axis_marker,
            obb_base=goal_info.get("obb"),
            standoff_base=(goal_x_base, goal_y_base),
            owner_map=(owner_map_x, owner_map_y),
            goal_map=(goal_x, goal_y),
            pose_map=(x0, y0, yaw0),
            owner_distance=distance,
            standoff_distance=standoff,
            note="goal standoff %.2fm" % standoff,
        )
        # ---- 中间点：正中一个候选 + 被障碍挡住时"换一侧"的候选 -----------------
        vector_x = goal_x_base - owner_x
        vector_y = goal_y_base - owner_y
        norm = max(1e-6, math.hypot(vector_x, vector_y))
        perp_x, perp_y = -vector_y / norm, vector_x / norm
        offsets = [0.0]
        for index in range(self.fall_nav_obstacle_retries):
            magnitude = self.fall_nav_obstacle_shift * (index + 1)
            offsets.extend([magnitude, -magnitude])

        ok = False
        reason = "未尝试"
        for attempt, offset in enumerate(offsets):
            try_x_base = goal_x_base + perp_x * offset
            try_y_base = goal_y_base + perp_y * offset
            goal_x = x0 + math.cos(yaw0) * try_x_base - math.sin(yaw0) * try_y_base
            goal_y = y0 + math.sin(yaw0) * try_x_base + math.cos(yaw0) * try_y_base
            if attempt:
                self._fall_console(
                    "%s：中间点被挡，第 %d 次尝试（换一侧，横移 %.2f m）"
                    % (label, attempt + 1, offset)
                )
                self.publish_fall_debug_markers(
                    label,
                    owner_base=(owner_x, owner_y),
                    samples_base=snapshot.get("points_xy"),
                    body_base=snapshot.get("body_xy"),
                    axis_base=axis_marker,
                    obb_base=goal_info.get("obb"),
                    standoff_base=(try_x_base, try_y_base),
                    owner_map=(owner_map_x, owner_map_y),
                    goal_map=(goal_x, goal_y),
                    pose_map=(x0, y0, yaw0),
                    owner_distance=distance,
                    standoff_distance=standoff,
                    note="stage %.2fm shift %+.2f" % (standoff, offset),
                )
            plan_ok, plan_reason = self.fall_nav_plan_ok((x0, y0), (goal_x, goal_y))
            if not plan_ok and self.fall_nav_plan_block:
                rospy.logwarn("%s：路径预检没过（%s）", label, plan_reason)
                self._fall_console("%s：路径预检没过（%s）" % (label, plan_reason))
                reason = "路径预检没过"
                continue
            if not plan_ok:
                # 现场要求：有障碍就交给 move_base 自己绕（预检只是提示，不是门槛）
                rospy.loginfo("%s：路径预检没过（%s），仍交给 move_base 尝试", label, plan_reason)
                self._fall_console("%s：预检没过（%s）但按现场要求仍交给 move_base 尝试" % (label, plan_reason))
            self._fall_console(
                "%s：中间点 map=(%.2f, %.2f)（%s）" % (label, goal_x, goal_y, plan_reason)
            )
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = self.fall_nav_frame
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = goal_x
            goal.target_pose.pose.position.y = goal_y
            goal.target_pose.pose.position.z = 0.0
            goal.target_pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
            goal.target_pose.pose.orientation.w = math.cos(goal_yaw / 2.0)
            ok, reason = self.navigate_fall_goal_with_guard(
                goal, self.fall_nav_timeout, label, owner_bbox=snapshot.get("bbox")
            )
            self._fall_console(
                "%s：move_base 结果 ok=%s（%s）" % (label, ok, reason or "到达")
            )
            if ok:
                break
            # 只有"换个位置有望解决"的失败才换一侧重试：
            #   有障碍  → 这一侧被挡；
            #   卡住    → 规划器进不去/出不来（看护循环新加的判据，2026-10-16）；
            #   还差    → move_base 假成功，换个落点再试。
            # 超时、ROS 关闭这类换哪一侧都一样，直接退下一级。
            if not any(key in str(reason) for key in ("有障碍", "卡住", "还差")):
                break
        if not ok:
            rospy.logwarn("%s：move_base 没走到中间点（%s），退回下一级", label, reason)
            return False
        rospy.loginfo("%s：已到中间点（≈%.2f m），开始直行闭环", label, standoff)
        return self.fall_finish_approach(snapshot.get("bbox"), label)

    # ------------------------------------------------------------------
    # 挥手求助靠近：B-2 式（导航段走代价地图 + 收尾段带雷达护栏）
    # ------------------------------------------------------------------
    @staticmethod
    def _approach_console(message):
        """挥手靠近的"终端可见"日志（print 到 stdout）。

        动作识别一触发，任务节点的 rosout/日志文件就会失灵（2026-10-11、10-16
        两次现场都遇到），关键步骤只有 stdout 一定看得见。
        """
        try:
            print("[挥手靠近] %s" % message, flush=True)
        except Exception:  # noqa: BLE001
            pass

    def approach_owner_navigate(self, owner_name="", initial_bbox=None):
        """挥手求助的"走近"：先 move_base 走到主人前方 standoff 点，再点云闭环收尾。

        为什么改成这样（2026-10-16 现场）：
          老的 approach_owner 是"相机闭环 + 直接发 /cmd_vel"—— 既不查 /scan，
          也不经过 move_base/代价地图。它的控制目标是"人到镜头的距离"，
          所以主人站在椅子后面挥手时，机器人盯着 1.0 m 这个数一路推过去，
          中间那把椅子腿在控制环里根本不存在，于是撞上去。
          （`fall_scan_guard_distance` 那个"硬护栏"只接在摔倒靠近的三个函数里，
           对挥手靠近是死代码；而且窗口只有正前 ±25°、阈值还是从雷达中心算的。）

        现在这条路和摔倒靠近一样：
          ① 点云快照出主人在机器人系的位置（躯干锚点 + 地面剔除）；
          ② 在"主人→机器人"连线上退 standoff（默认 approach_standoff_distance
             = 1.0 m）得到中间点，换算到 map 系交给 move_base —— **走代价地图**，
             椅子腿、桌腿这些障碍由全局/局部规划器绕开；
          ③ 导航途中双保险：move_base 自己的避障 + `/scan` 硬护栏
             （含"挡在前面的到底是不是主人"的判别，见 guard_obstacle_is_owner）；
          ④ 到中间点后，最后一段用点云闭环 `drive_forward_until_cloud_close`
             收到 approach_standoff_distance，这一段同样带 /scan 护栏。

        返回 True=已经站到主人前方合适距离；False=没走通（调用方按
        ~approach_nav_fallback_camera 决定是原地不动还是退回老办法）。
        """
        if not self.approach_enabled:
            rospy.loginfo("走近主人已关闭（approach_enabled=false）")
            return False
        if not self.approach_navigate_enabled:
            return False

        target = float(self.approach_standoff_distance)
        label = "%s（挥手·导航靠近）" % (owner_name or "主人")
        stage = float(self.approach_nav_stage_standoff_distance)
        standoff = target if stage <= 0.0 else max(0.50, stage)
        self._approach_console("%s：开始（目标 %.2f m，中间点 %.2f m）" % (label, target, standoff))

        if not self.ensure_approach_pointcloud():
            rospy.logwarn("%s：点云不可用，不走导航方式", label)
            self._approach_console("%s：点云不可用 → 不走导航方式" % label)
            return False
        self.ensure_scan_subscription()
        band = (self.approach_owner_band_low, self.approach_owner_band_high)
        snapshot = self.capture_fallen_owner_snapshot(
            initial_bbox, label=label, band=band, floor_filter=True
        )
        if snapshot is None:
            rospy.logwarn("%s：拿不到主人的点云位置，不走导航方式", label)
            self._approach_console("%s：快照失败 → 不走导航方式" % label)
            return False
        # "我这就过来"整次靠近只播一次（与摔倒靠近共用同一个标志位）
        self.fall_speak_approach_start_once()
        distance = float(snapshot["distance"])
        rospy.loginfo(
            "%s：快照 前=%.2f m 侧=%.2f m 距离=%.2f m bearing=%.3f rad"
            "（样本 %d 次 / %d 点）",
            label,
            snapshot["forward"],
            snapshot["lateral"],
            distance,
            float(snapshot.get("bearing") or 0.0),
            snapshot["samples"],
            snapshot["points"],
        )
        self._approach_console(
            "%s：快照 前=%.2f 侧=%.2f 距离=%.2f bearing=%.3f（%d 点）"
            % (
                label,
                snapshot["forward"],
                snapshot["lateral"],
                distance,
                float(snapshot.get("bearing") or 0.0),
                int(snapshot.get("points") or 0),
            )
        )

        # 可选：先原地"裸转正"。默认关 —— 裸转没有护栏（rotate_toward_bearing 只看
        # 里程计），交给 move_base 去转到目标朝向更安全；face-first 只作为现场可选。
        bearing = float(snapshot.get("bearing") or 0.0)
        if self.approach_face_first_enabled and abs(bearing) > self.approach_face_first_max_angle:
            self._approach_console("%s：按现场开关做转正（bearing=%.3f rad）" % (label, bearing))
            self.rotate_toward_bearing(bearing, label=label)

        owner_x = float(snapshot["forward"]) + self.approach_camera_forward_offset
        owner_y = float(snapshot["lateral"])
        # 站姿主人不传 body_xy：身体主方向是给"躺着的人"用的，站姿下它没有意义，
        # 传 None 就是沿视线退 standoff（正对主人停下）。
        goal_x_base, goal_y_base, face_yaw_rel, goal_info = self.fall_standoff_goal_base(
            owner_x, owner_y, standoff, body_xy=None, label=label
        )

        # 已经站在中间点附近：不导航，直接进"点云闭环收尾"
        if distance - standoff <= self.fall_nav_distance_tolerance:
            rospy.loginfo(
                "%s：已在中间点附近（%.2f m ≤ %.2f + %.2f），跳过导航直接收尾",
                label,
                distance,
                standoff,
                self.fall_nav_distance_tolerance,
            )
            self._approach_console("%s：已在中间点附近（%.2f m），跳过导航" % (label, distance))
            return self.approach_finish_by_cloud(snapshot.get("bbox"), label)

        pose = self.robot_pose_in_map()
        if pose[0] is None:
            rospy.logwarn("%s：拿不到 map 系位姿（TF 不通），不走导航方式", label)
            self._approach_console("%s：拿不到 map 位姿（TF 不通）" % label)
            return False
        x0, y0, yaw0, age = pose
        goal_x = x0 + math.cos(yaw0) * goal_x_base - math.sin(yaw0) * goal_y_base
        goal_y = y0 + math.sin(yaw0) * goal_x_base + math.cos(yaw0) * goal_y_base
        goal_yaw = self.wrap_angle(yaw0 + float(face_yaw_rel))
        owner_map_x = x0 + math.cos(yaw0) * owner_x - math.sin(yaw0) * owner_y
        owner_map_y = y0 + math.sin(yaw0) * owner_x + math.cos(yaw0) * owner_y
        rospy.loginfo(
            "%s：主人 map≈(%.2f, %.2f)，中间点 map=(%.2f, %.2f) yaw=%.2f；"
            "机器人 map=(%.2f, %.2f) yaw=%.2f（位姿 age=%.1fs）",
            label,
            owner_map_x,
            owner_map_y,
            goal_x,
            goal_y,
            goal_yaw,
            x0,
            y0,
            yaw0,
            age if age is not None else -1.0,
        )
        self._approach_console(
            "%s：主人 map≈(%.2f, %.2f) 中间点 map=(%.2f, %.2f)"
            % (label, owner_map_x, owner_map_y, goal_x, goal_y)
        )
        # 调试可视化（只画图，不影响决策）：红=主人估计点、绿=中间点、蓝=机器人
        self.publish_fall_debug_markers(
            label,
            owner_base=(owner_x, owner_y),
            samples_base=snapshot.get("points_xy"),
            body_base=snapshot.get("body_xy"),
            standoff_base=(goal_x_base, goal_y_base),
            owner_map=(owner_map_x, owner_map_y),
            goal_map=(goal_x, goal_y),
            pose_map=(x0, y0, yaw0),
            owner_distance=distance,
            standoff_distance=standoff,
            note="approach standoff %.2fm" % standoff,
        )

        # ---- 中间点：正中一个候选 + 被挡时"换一侧"的候选 ----
        vector_x = goal_x_base - owner_x
        vector_y = goal_y_base - owner_y
        norm = max(1e-6, math.hypot(vector_x, vector_y))
        perp_x, perp_y = -vector_y / norm, vector_x / norm
        offsets = [0.0]
        for index in range(self.approach_nav_obstacle_retries):
            magnitude = self.approach_nav_obstacle_shift * (index + 1)
            offsets.extend([magnitude, -magnitude])

        ok = False
        reason = "未尝试"
        for attempt, offset in enumerate(offsets):
            try_x_base = goal_x_base + perp_x * offset
            try_y_base = goal_y_base + perp_y * offset
            goal_x = x0 + math.cos(yaw0) * try_x_base - math.sin(yaw0) * try_y_base
            goal_y = y0 + math.sin(yaw0) * try_x_base + math.cos(yaw0) * try_y_base
            if attempt:
                self._approach_console(
                    "%s：中间点被挡，第 %d 次尝试（换一侧，横移 %.2f m）"
                    % (label, attempt + 1, offset)
                )
            plan_ok, plan_reason = self.fall_nav_plan_ok((x0, y0), (goal_x, goal_y))
            if not plan_ok and self.fall_nav_plan_block:
                rospy.logwarn("%s：路径预检没过（%s）", label, plan_reason)
                self._approach_console("%s：路径预检没过（%s）" % (label, plan_reason))
                reason = "路径预检没过"
                continue
            if not plan_ok:
                rospy.loginfo("%s：路径预检没过（%s），仍交给 move_base 尝试", label, plan_reason)
            self._approach_console(
                "%s：中间点 map=(%.2f, %.2f)（%s）" % (label, goal_x, goal_y, plan_reason)
            )
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = self.fall_nav_frame
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = goal_x
            goal.target_pose.pose.position.y = goal_y
            goal.target_pose.pose.position.z = 0.0
            goal.target_pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
            goal.target_pose.pose.orientation.w = math.cos(goal_yaw / 2.0)
            ok, reason = self.navigate_fall_goal_with_guard(
                goal, self.approach_nav_timeout, label, owner_bbox=snapshot.get("bbox")
            )
            self._approach_console("%s：move_base 结果 ok=%s（%s）" % (label, ok, reason or "到达"))
            if ok:
                break
            # 只有"换一侧有望解决"的失败才重试：有障碍 / 卡住 / 还差
            if not any(key in str(reason) for key in ("有障碍", "卡住", "还差")):
                break
        if not ok:
            rospy.logwarn("%s：move_base 没走到中间点（%s）", label, reason)
            self._approach_console("%s：move_base 没走到中间点（%s）" % (label, reason))
            return False
        rospy.loginfo("%s：已到中间点（≈%.2f m），开始点云闭环收尾", label, standoff)
        self._approach_console("%s：已到中间点（≈%.2f m），开始收尾" % (label, standoff))
        return self.approach_finish_by_cloud(snapshot.get("bbox"), label)

    def approach_finish_by_cloud(self, bbox, label):
        """导航段之后的收尾：用**点云闭环**收到 approach_standoff_distance。

        2026-10-16 现场要求：**挥手靠近要减少最后的"盲靠"**（纯里程计往前贴）。
        原因：站姿主人 1 m 内点云照样量得到，根本没有"非得盲走"的理由；而盲走
        没有任何相机/点云确认，只有正前方 ±25°、0.40 m 的雷达护栏，正是撞椅子腿
        那一类风险。所以这里改成：

          · 先跑点云闭环（`drive_forward_until_cloud_close`，带 /scan 护栏）；
          · 还差一点就**再用闭环补**（`~approach_finish_retries`，默认 1 次），
            而不是盲走；
          · 只有显式打开 `~approach_creep_max_distance`(默认 0 = 关) 时才允许
            里程计补齐，而且照样带 /scan 护栏。

        返回 True=已确认站到 standoff 附近；False=没确认（调用方直接进入问答，
        不再前进）。
        """
        target = float(self.approach_standoff_distance)
        tolerance = max(0.05, float(self.fall_nav_distance_tolerance))
        attempts = 1 + max(0, int(self.approach_finish_retries))
        measured = None
        reason = "未尝试"
        for attempt in range(attempts):
            measured, reason = self.drive_forward_until_cloud_close(
                bbox, label=label, stop_distance=target
            )
            self._approach_console(
                "%s：收尾第 %d/%d 次闭环结束（%s，点云距离=%s，目标 %.2f m）"
                % (
                    label,
                    attempt + 1,
                    attempts,
                    reason,
                    "无" if measured is None else "%.2f m" % float(measured),
                    target,
                )
            )
            if reason == "scan_guard":
                # 护栏就是"前面有东西挡着"：不再前进，也不盲开
                rospy.logwarn("%s：前方有障碍（护栏触发），停在原地不再靠近", label)
                return False
            if measured is None:
                # 量不到人（多半已经太近/出画面）：再往前就是盲开，停
                rospy.loginfo("%s：收尾时量不到人，停在这个位置（不盲走）", label)
                break
            if float(measured) <= target + tolerance:
                break
            if attempt + 1 < attempts:
                rospy.loginfo(
                    "%s：还差 %.2f m（点云 %.2f m > 目标 %.2f m + 容差 %.2f），"
                    "用点云闭环再补一次（不盲走）",
                    label,
                    float(measured) - target,
                    float(measured),
                    target,
                    tolerance,
                )
        arrived = measured is not None and float(measured) <= target + tolerance
        # 可选：显式打开才做里程计补齐（默认关）。留这个口子是为了极端场景下手动兜底。
        if (
            not arrived
            and measured is not None
            and self.approach_creep_max_distance > 0.0
        ):
            remaining = max(0.0, float(measured) - target)
            remaining = min(self.approach_creep_max_distance, remaining)
            if remaining > 0.15:
                done, travelled = self.creep_forward_by_odometry(
                    remaining,
                    label=label,
                    scan_stop_distance=self.fall_scan_guard_distance,
                )
                rospy.loginfo(
                    "%s：按开关做里程计补齐 %.2f m / %.2f m 完成=%s（雷达护栏 %.2f m）",
                    label,
                    travelled,
                    remaining,
                    done,
                    self.fall_scan_guard_distance,
                )
                self._approach_console(
                    "%s：里程计补齐 %.2f m / %.2f m 完成=%s" % (label, travelled, remaining, done)
                )
        if not arrived:
            rospy.logwarn(
                "%s：没能确认走到 %.2f m（点云最后 %s，%s）→ 不再前进",
                label,
                target,
                "无" if measured is None else "%.2f m" % float(measured),
                reason,
            )
        return bool(arrived)

    def take_pending_answer(self):
        """取走并清空当前待处理的 ASR 文本（没有就返回空串）。"""
        with self.name_condition:
            answer = self.latest_name_answer
            self.latest_name_answer = None
        return answer or ""

    def listen_for_reply(self, timeout):
        """等主人说一句话，返回识别到的文本；超时返回 None。

        复用注册问答那套 ASR 通路（asr_callback 把结果塞进 latest_name_answer），
        只是把它"收答案"的开关打开 —— 这样回声过滤（is_likely_echo）、
        退出处理都不用重写一遍。

        为什么要分两段等：ASR 是**重叠窗口**发布的（约每 2 秒一条，先短后长，
        比如"帮我开" → "帮我开一下灯"）。拿到第一条就走的话，复述给评委听的
        就是被切掉半截的话。所以收到第一条后，再多收 ~listen_settle_seconds
        一窗，取其中**最长**的那条当最终答案。
        """
        with self.name_condition:
            self.latest_name_answer = None
            self.accepting_name_answer = True
        best = ""
        try:
            deadline = time.time() + max(0.5, float(timeout))
            # 第一段：等第一句
            while not rospy.is_shutdown() and time.time() < deadline:
                answer = self.take_pending_answer()
                if answer:
                    best = answer
                    break
                self.update_yolo_window("等待主人回答")
                rospy.sleep(0.2)
            # 第二段：再收一窗，把更完整的那一版也收进来
            if best and self.listen_settle_seconds > 0.0:
                settle_deadline = time.time() + self.listen_settle_seconds
                while not rospy.is_shutdown() and time.time() < settle_deadline:
                    answer = self.take_pending_answer()
                    if answer and len(answer) > len(best):
                        best = answer
                    self.update_yolo_window("等主人说完")
                    rospy.sleep(0.2)
            return best or None
        finally:
            with self.name_condition:
                self.accepting_name_answer = False

    def ask_owner_help(self, owner_name=""):
        """问主人需要什么帮助，并把需求**按原意复述**回去。返回原话（没有则 None）。

        规则 2.6-5 的判据是「能够将主人的帮助需求以原意复述」。旧实现只播一句
        "请问您需要什么帮助？"就算做完 —— 问了、但没听、也没复述，那 50 分拿不到。
        所以这里必须：真听（listen_for_reply）→ 用主人自己的话念回去。
        听不清就明说听不清，别装作懂了、更别自己编一句。
        """
        if not self.ask_help_enabled:
            rospy.loginfo("挥手求助对话已关闭（ask_help_enabled=false）")
            return None

        target_label = owner_name or "主人"
        answer = None
        for attempt in range(self.ask_help_retries + 1):
            prompt = (
                self.ask_help_question_text
                if attempt == 0
                else self.ask_help_retry_text
            )
            self.speak(prompt, wait=True)
            answer = self.listen_for_reply(self.ask_help_answer_timeout)
            if answer:
                break
            rospy.logwarn(
                "%s第 %d 轮没听到回答（共 %d 轮）",
                target_label,
                attempt + 1,
                self.ask_help_retries + 1,
            )
        if not answer:
            rospy.logwarn("问%s需要什么帮助：%d 轮都没听到回答",
                          target_label, self.ask_help_retries + 1)
            self.speak(self.ask_help_timeout_text, wait=True)
            self.publish_status(
                "owner_help_no_answer",
                owner_name=owner_name,
                timeout=self.ask_help_answer_timeout,
                attempts=self.ask_help_retries + 1,
            )
            return None

        speech = self.ask_help_retell_text % answer
        rospy.loginfo("复述%s的需求：%s", target_label, answer)
        self.speak(speech, wait=True)
        self.publish_status(
            "owner_help_request",
            owner_name=owner_name,
            request=answer,
        )
        return answer

    # ------------------------------------------------------------------
    # 播报后的实际反应（三）：电器开关（只做"听懂 + 播报"）
    # ------------------------------------------------------------------
    def switch_state_speech(self, intent):
        """电器开关的结果播报。文案统一放在 feedback_after_recognition.py。"""
        try:
            module = self.feedback_module()
            speech = module.switch_state_speech(intent)
            if speech:
                return speech
        except Exception as exc:
            rospy.logwarn("拼电器开关播报失败，退回默认说法：%s", exc)
        return {
            "switch_on": "好的，电器开关已经打开了。",
            "switch_off": "好的，电器开关已经关掉了。",
            "switch_declined": "好的，那我就不动电器开关了。",
        }.get(str(intent or "").strip().lower(), "")

    def switch_intent_of(self, transcript):
        """主人的一句话 -> switch_on / switch_off / switch_declined / None。"""
        try:
            return self.feedback_module().parse_switch_intent(transcript)
        except Exception as exc:
            rospy.logwarn("解析电器开关指令失败：%s", exc)
            return None

    def ask_switch_instruction(self, owner_name=""):
        """问主人要不要操作电器开关，听懂"要开 / 要关 / 不用"。返回意图名或 None。

        ⚠️ 2026-10-16 现场已**停用**：坐姿/躺着的 next_step 改成 "next_waypoint"，
        分派器不再调用本函数（也不再有"等主人回答"那一段）。
        实现原样保留，想恢复就把 feedback_after_recognition.ACTION_NEXT_STEP
        里 sitting/lying 改回 "ask_switch"。

        只问一轮：识别不到就明说没听清，不硬猜 —— 猜错了播报的内容就是错的，
        而这条的得分依据恰恰是播报内容。
        """
        if not self.ask_switch_enabled:
            rospy.loginfo("电器开关询问已关闭（ask_switch_enabled=false）")
            return None

        target_label = owner_name or "主人"
        self.speak(self.ask_switch_question_text, wait=True)
        answer = self.listen_for_reply(self.ask_switch_answer_timeout)
        intent = self.switch_intent_of(answer) if answer else None
        if intent is None:
            rospy.logwarn(
                "问%s要不要操作电器开关：没听出「要开还是要关」（听到的是 %r）",
                target_label,
                answer or "",
            )
            self.speak(self.ask_switch_no_answer_text, wait=True)
            self.publish_status(
                "switch_instruction_unknown",
                owner_name=owner_name,
                transcript=answer or "",
            )
            return None

        rospy.loginfo(
            "听懂%s对电器开关的要求：%s（原话：%s）", target_label, intent, answer
        )
        self.publish_status(
            "switch_instruction",
            owner_name=owner_name,
            intent=intent,
            transcript=answer,
        )
        return intent

    def operate_switch(self, owner_name="", intent=None):
        """电器开关：目前**只播报状态，不动机械臂**。

        为什么不动机械臂：这套动作要靠"开关标记视觉检测 + 机械臂对位"才能把末端
        停到"开/关"标记正上方，而检测用的硬件暂时不支持。与其让机械臂乱比划，
        不如老老实实把开关的状态用话说清楚 —— 判分依据本来就是语音播报。

        等硬件和标记到位，把下面的播报**前后**各补一段机械臂对位即可
        （参考旧包的 perform_fall_assist_arm_motion：往 /wpb_home/mani_ctrl
         发 lift / gripper 的 JointState），主流程和 dispatch_next_step 不用改。
        """
        target_label = owner_name or "主人"
        if intent is None:
            rospy.logwarn("要播报电器开关状态，但没拿到「要开还是要关」，跳过")
            return None
        speech = self.switch_state_speech(intent)
        if not speech:
            rospy.logwarn("电器开关意图 %r 没有对应的播报文案，跳过", intent)
            return None
        rospy.loginfo("电器开关（仅语音播报）：%s -> %s", target_label, speech)
        self.speak(speech, wait=True)
        self.publish_status(
            "switch_operated",
            owner_name=owner_name,
            intent=intent,
            speech=speech,
            hardware="voice_only",
        )
        return speech

    # ------------------------------------------------------------------
    # 预留接口：还没实现的功能都在这里占位
    # ------------------------------------------------------------------
    # 说明：这些方法现在只打日志、返回 None，主流程调用它们不会出错。
    # 以后实现时**只改这些方法内部**，主流程（run / check_found_owner）不用动。
    # 分派入口是 dispatch_next_step()，它按 feedback_after_recognition.py 给出的
    # next_step 名字路由，名字是两边约定的契约。
    # 已经做完的：navigate_to_waypoint / return_to_exit（导航）
    #             approach_owner / ask_owner_help（走近 + 挥手求助对话）
    #             ask_switch_instruction / operate_switch（电器开关：只做"听懂 + 播报"）
    #             perform_fall_assist（摔倒协助：走近 + 伸机械臂）
    # 还没做的：电器开关里"开关标记视觉检测 + 机械臂对位"那半 —— 硬件暂时不支持，
    #           所以 operate_switch 现在只播报状态，不发任何机械臂指令。
    #           check_hardware（开机硬件自检）仍是占位。
    def is_topic_advertised(self, topic_name):
        """话题有没有发布者。查询失败返回 None（不把自检卡死）。"""
        try:
            published_topics = rospy.get_published_topics("/")
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn("查询话题列表失败（自检降级为只看有没有数据）：%s", exc)
            return None
        normalized = str(topic_name).rstrip("/") or "/"
        for name, _topic_type in published_topics or []:
            if (str(name).rstrip("/") or "/") == normalized:
                return True
        return False

    def describe_missing_topic(self, topic_name, label):
        """把"缺哪一路数据"说成人话，直接给出下一步该跑什么命令。"""
        advertised = self.is_topic_advertised(topic_name)
        if advertised is True:
            return (
                "%s（%s）：有发布者但没有数据 → 跑 `rostopic hz %s` 确认，"
                "驱动起来了但没出数据（USB 掉了 / 设备被占用）"
                % (label, topic_name, topic_name)
            )
        if advertised is False:
            return (
                "%s（%s）：连发布者都没有 → 起栈缺了对应驱动"
                % (label, topic_name)
            )
        return "%s（%s）：没有收到数据" % (label, topic_name)

    def check_hardware(self):
        """开机硬件自检：等 /scan 和 /odom 真的出数据，缺哪个就报出来并停车。

        2026-10-16 加，移植自 server_robot3 的 wait_for_hardware_inputs()。
        返回 True=可以开始任务；False=硬件没就绪，调用方必须直接结束（不要开车）。

        相机不在这里等：本包的 wait_for_camera_inputs() 已经单独把关（它还会
        提示看门狗的自愈进度），这里只补上以前完全没人管的 /scan 和 /odom。
        """
        if not self.require_scan and not self.require_odom:
            rospy.loginfo(
                "硬件自检跳过（_require_scan=false _require_odom=false）"
            )
            return True
        if self.require_scan:
            self.ensure_scan_subscription()

        started = time.time()
        deadline = started + self.hardware_check_timeout
        last_report = 0.0
        rate = rospy.Rate(10)
        scan_ok = not self.require_scan
        odom_ok = not self.require_odom
        scan_age = None
        odom_age = None
        while not rospy.is_shutdown():
            now = time.time()
            with self.lock:
                scan_stamp = float(getattr(self, "_scan_msg_stamp", 0.0) or 0.0)
                odom_stamp = float(getattr(self, "latest_odom_stamp", 0.0) or 0.0)
            scan_age = (now - scan_stamp) if scan_stamp > 0.0 else None
            odom_age = (now - odom_stamp) if odom_stamp > 0.0 else None
            scan_ok = (not self.require_scan) or (
                scan_age is not None and scan_age <= 2.0
            )
            odom_ok = (not self.require_odom) or (
                odom_age is not None and odom_age <= 2.0
            )
            if scan_ok and odom_ok:
                rospy.loginfo(
                    "硬件自检通过：%s age=%s，%s age=%s（%.1f 秒）",
                    self.scan_topic,
                    "未要求" if scan_age is None else "%.2fs" % scan_age,
                    self.odom_topic,
                    "未要求" if odom_age is None else "%.2fs" % odom_age,
                    now - started,
                )
                return True
            if now >= deadline:
                break
            if now - last_report >= 5.0:
                last_report = now
                rospy.logwarn(
                    "等待硬件数据：%s %s / %s %s（已等 %.0f / %.0f 秒）",
                    self.scan_topic,
                    "有" if scan_ok else "无",
                    self.odom_topic,
                    "有" if odom_ok else "无",
                    now - started,
                    self.hardware_check_timeout,
                )
            rate.sleep()

        missing = []
        if not scan_ok:
            missing.append(self.describe_missing_topic(self.scan_topic, "雷达"))
        if not odom_ok:
            missing.append(self.describe_missing_topic(self.odom_topic, "底盘里程计"))
        rospy.logerr(
            "硬件自检没通过，任务不启动（避免盲开）：%s",
            "；".join(missing) or "未知",
        )
        rospy.logerr(
            "排查顺序：① 起栈是否完整；② `rostopic hz %s` 和 `rostopic hz %s`；"
            "③ 雷达 USB（`ls -l /dev/rplidar`）和底盘串口（`ls -l /dev/ftdi`）；"
            "④ 确实要先跳过自检时加参数 `_require_scan:=false _require_odom:=false`",
            self.scan_topic,
            self.odom_topic,
        )
        self.stop_base()
        return False

    def perform_fall_assist(self, owner_name="", action=""):
        """摔倒协助：把机械臂伸到人上方，保持几秒，再收回。

        动作来自旧包同一台机器上实测用过的那一套（`perform_fall_assist_arm_motion`）：
        往 /wpb_home/mani_ctrl 反复发 lift/gripper 的 JointState，
          伸出去（lift=0.50, gripper=0.12）-> 保持 4 秒 -> 收回（lift=0.0, gripper=0.12）
        每条指令都要**连续发**几秒，因为机械臂控制器只认"持续收到目标"，
        发一次就停的话它可能在半路停下（旧包注释里写过这一点）。
        """
        if not self.fall_assist_enabled:
            rospy.loginfo("摔倒协助已关闭（fall_assist_enabled=false）")
            return None
        target_label = owner_name or "主人"
        if not self.fall_assist_arm_enabled:
            rospy.loginfo("摔倒协助：机械臂动作已关闭，只走近不伸臂")
            self.speak("我过来了，请慢一点起来。", wait=True)
            return None

        if self.mani_ctrl_pub is None:
            try:
                self.mani_ctrl_pub = rospy.Publisher(
                    self.mani_ctrl_topic, JointState, queue_size=5
                )
                rospy.sleep(0.3)  # 等控制器订阅上，否则第一条指令会丢
            except Exception as exc:
                rospy.logwarn("创建机械臂发布器失败（%s）：%s", self.mani_ctrl_topic, exc)
                return None

        rospy.loginfo(
            "摔倒协助（%s 动作=%s）：伸臂 lift=%.2f gripper=%.2f 保持 %.1fs，"
            "再收回 lift=%.2f gripper=%.2f",
            target_label,
            action or "?",
            self.fall_assist_arm_extend_lift,
            self.fall_assist_arm_extend_gripper,
            self.fall_assist_arm_hold_seconds,
            self.fall_assist_arm_retract_lift,
            self.fall_assist_arm_retract_gripper,
        )
        self.publish_status(
            "fall_assist_started",
            owner_name=owner_name,
            action=action,
            standoff=self.fall_approach_standoff_distance,
        )
        self.hold_manipulator_command(
            self.fall_assist_arm_extend_lift,
            self.fall_assist_arm_extend_gripper,
            self.fall_assist_arm_extend_wait + self.fall_assist_arm_hold_seconds,
        )
        self.hold_manipulator_command(
            self.fall_assist_arm_retract_lift,
            self.fall_assist_arm_retract_gripper,
            self.fall_assist_arm_retract_wait,
        )
        if self.fall_assist_arm_completion_wait > 0.0:
            rospy.sleep(self.fall_assist_arm_completion_wait)
        if str(self.fall_assist_done_text or "").strip():
            self.speak(self.fall_assist_done_text, wait=True)
        self.publish_status(
            "fall_assist_finished",
            owner_name=owner_name,
            action=action,
        )
        return True

    def publish_manipulator_command(self, lift, gripper):
        """发一条机械臂目标（lift/gripper）。"""
        if self.mani_ctrl_pub is None:
            return False
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["lift", "gripper"]
        message.position = [float(lift), float(gripper)]
        message.velocity = [
            float(self.fall_assist_arm_lift_velocity),
            float(self.fall_assist_arm_gripper_velocity),
        ]
        self.mani_ctrl_pub.publish(message)
        return True

    def hold_manipulator_command(self, lift, gripper, seconds):
        """把同一个目标连续发 seconds 秒（机械臂只认持续收到的目标）。"""
        deadline = time.time() + max(0.0, float(seconds))
        rate = rospy.Rate(self.fall_assist_arm_command_rate)
        self.publish_manipulator_command(lift, gripper)
        while not rospy.is_shutdown() and time.time() < deadline:
            self.publish_manipulator_command(lift, gripper)
            rate.sleep()

    def return_to_exit(self):
        """全部完成后导航回出口（自主离场）。

        这是"最后一步"，时间预算里预留的 ~competition_exit_reserve_seconds 就是
        留给它的。不导航模式（navigate_enabled=false）下只打日志，不动。
        """
        if not self.navigate_enabled or not self.return_to_exit_enabled:
            rospy.loginfo("返回出口：未启用（导航=%s 开关=%s）",
                          "开" if self.navigate_enabled else "关",
                          "开" if self.return_to_exit_enabled else "关")
            return None
        if not self.exit_waypoint_name:
            rospy.loginfo("返回出口：没有配置出口航点，跳过")
            return None
        self.log_time_budget("准备返回出口")
        self.speak("任务完成，正在返回出口。", wait=True)
        # 回出口是"新目标"，重新计数：别让前面房间的失败把最后这段也掐掉
        self.reset_room_navigation_failures(self.exit_waypoint_name)
        arrived = self.navigate_to_waypoint(self.exit_waypoint_name)
        if arrived:
            rospy.loginfo("已返回出口航点 %s", self.exit_waypoint_name)
        else:
            rospy.logwarn("返回出口失败/跳过（航点 %s）", self.exit_waypoint_name)
        return arrived

    def dispatch_next_step(self, next_step, owner_name="", action="", owner_bbox=None):
        """真正的分派入口：外面包一层异常保护。

        "走近/摔倒协助"这类动作里任何异常都不许把主流程带崩 —— 现场踩过：
        异常之后任务一声不吭地卡住，看起来就是"检测到摔倒却什么都没做"。
        """
        try:
            return self._dispatch_next_step(
                next_step, owner_name, action, owner_bbox
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(
                "[摔倒靠近] 下一步动作 %s 执行异常：%s" % (next_step, exc),
                flush=True,
            )
            traceback.print_exc()
            rospy.logerr("下一步动作 %s 执行异常（已忽略，流程继续）：%s", next_step, exc)
            return None

    def _dispatch_next_step(self, next_step, owner_name="", action="", owner_bbox=None):
        """按"下一步动作名"分派到对应模块。

        next_step 的取值由 feedback_after_recognition.py 定义（两边约定的契约）：
          approach_and_assist  -> 走近 + 摔倒机械臂协助
          approach_and_ask_help-> 走近 + 问需要什么帮助
          ask_switch           -> 问电器开关指令
          operate_switch       -> 执行电器开关
          retry / unknown      -> 什么都不做
        以后新增能力：在这里加一个分支 + 实现对应方法即可。
        """
        rospy.loginfo("下一步动作：%s（主人=%s 动作=%s）", next_step, owner_name, action)
        if next_step == "next_waypoint":
            # 2026-10-16 现场要求：坐姿 / 躺在床上，播报完"动作+位置"就**直接继续**，
            # 不追问"要不要操作电器开关"、也不等主人回答（原来这一步要停 8~15 秒）。
            rospy.loginfo(
                "按现场要求：坐姿/躺着不追问电器开关，直接继续下一个航点"
            )
            return None
        if next_step == "approach_and_assist":
            # 摔倒协助：走近 → 伸机械臂 → 收回。
            # 整次靠近只播一次"我这就过来"（B-2/B-1/相机闭环三级回退不再各播一遍）
            self._fall_approach_announced = False
            # 现在敢对躺着的人调用 approach_owner 了，因为距离是**点云量出来的**：
            # 躺姿的人体框又宽又矮，"框高占比"那条路会以为还很远、一路开到人身上；
            # 点云量的是人到相机的真实深度，跟姿势无关。
            # （点云不可用时 approach_owner 会退回框高占比 —— 那种情况下这条
            #   会停在偏远处，比开过头安全，日志里会写明是估计值。）
            mode = str(self.fall_approach_mode or "camera").strip().lower()
            if mode == "navigate":
                # B-2：快照 3D 位置 → map 系 standoff 目标交 move_base（走 costmap
                # 绕障）→ 到位后用里程计补齐最后 0.25 m。没走通就逐级退回。
                if not self.approach_fallen_owner_navigate(owner_name, owner_bbox):
                    rospy.logwarn("摔倒靠近（导航式）没走通，退回快照式（B-1）")
                    mode = "snapshot"
            if mode == "snapshot":
                # B-1：快照 3D 位置 → 转正 → 直行到云快看不见 → 里程计爬完最后 0.5 m
                if not self.approach_fallen_owner_snapshot(owner_name, owner_bbox):
                    rospy.logwarn("摔倒靠近（快照式）未完成，退回相机闭环")
                    mode = "camera"
            if mode == "camera":
                if not getattr(self, "fall_camera_fallback_enabled", False):
                    # 2026-10-12 实地要求：导航段没能把机器人带到中间点（例如周围全是
                    # 障碍物、真走不过去）就**别再折腾**——原地停稳，直接做伸臂动作。
                    rospy.logwarn(
                        "摔倒靠近：导航/快照式都没走通，按要求不再退相机闭环，"
                        "原地停下直接伸臂协助"
                    )
                    self._fall_console("无法再靠近（导航走不过去）→ 原地停下，直接伸臂")
                    self.stop_base()
                else:
                    self.approach_owner(
                        owner_name,
                        standoff_distance=self.fall_approach_standoff_distance,
                        min_distance=self.fall_approach_min_distance,
                        initial_bbox=owner_bbox,
                        lost_seconds=self.fall_approach_lost_seconds,
                        identity_gate=False,
                    )
            return self.perform_fall_assist(owner_name, action)
        if next_step == "approach_and_ask_help":
            # 挥手求助：走近 → 问需要什么帮助 → 把主人的原话复述回去
            # 走近从 2026-10-16 起走"导航式"：先 move_base（走代价地图，绕开椅子腿等
            # 障碍；导航段带 /scan 硬护栏），最后一段用点云闭环 + /scan 护栏收尾。
            # 走不通时默认**原地不动**直接问答，不退回那条没有护栏的相机闭环
            # （现场就是在那条路上撞的椅子腿）。想退回老行为：_approach_navigate_fallback_camera:=true
            self._fall_approach_announced = False
            if not self.approach_enabled:
                rospy.loginfo("走近主人已关闭（approach_enabled=false），原地直接进入问答")
            elif self.approach_owner_navigate(owner_name, owner_bbox):
                self._approach_console("挥手靠近：已站到主人前方合适距离")
            elif self.approach_navigate_fallback_camera:
                rospy.logwarn("挥手靠近：导航方式没走通，按开关退回相机闭环（**无雷达护栏**）")
                self._approach_console("挥手靠近：导航没走通 → 退回相机闭环（无护栏）")
                self.approach_owner(owner_name, initial_bbox=owner_bbox)
            else:
                rospy.logwarn(
                    "挥手靠近：没能确认走到 standoff，不再前进、直接进入问答"
                    "（避免撞到周围家具/椅子腿）"
                )
                self._approach_console("挥手靠近：没走到位 → 不再前进，直接问答")
            return self.ask_owner_help(owner_name)
        if next_step == "ask_switch":
            # 问 → 听懂要开还是要关 → 把开关状态播报出来（机械臂那半还没接）
            intent = self.ask_switch_instruction(owner_name)
            if intent in ("switch_on", "switch_off"):
                return self.operate_switch(owner_name, intent)
            return None
        if next_step == "operate_switch":
            # 这条来自"语音指令"那条路（feedback_for_voice 给出 intent）。
            # 语音意图识别还没接进来，走到这里时手上没有明确的意图 ——
            # 不猜（猜错就是播报错），直接跳过。
            rospy.loginfo("operate_switch 分支需要明确的开/关意图，当前没有，跳过")
            return None
        return None

    # ------------------------------------------------------------------
    # 逐位寻找 + 认人 + 报动作
    # ------------------------------------------------------------------
    def run_action_recognition_for_owner(self, owner_result):
        """跑一次动作识别。

        父类的 run_owner_action_recognition 带"只跑一次"的锁
        （action_completed），那是给"找到一位就结束"的老流程用的。
        这里每位主人各跑一次，所以每次调用前把锁和上一次的结果清干净。
        """
        self.action_completed = False
        self.action_result = None
        if hasattr(self, "action_result_event"):
            self.action_result_event.clear()
        # 动作识别期间把垃圾检测暂停：两个都在同一个 CPU 上跑，抢起来会拖慢
        # 动作识别（那是要对着主人现场出结果的），而这段时间镜头正对着主人，
        # 垃圾也不在视野里 —— 这 20~30 秒不看垃圾不亏。
        self.set_trash_detector_paused(True, "动作识别期间")
        try:
            return self.run_owner_action_recognition(owner_result)
        finally:
            self.set_trash_detector_paused(False, "动作识别结束，接着找垃圾")

    def find_and_check_owner(self, owner_profile):
        """找这一位主人 → 居中 → 认人 → 检测动作。

        返回 dict：找到的信息 + 动作结果；没找到返回 None。
        """
        owner_index = owner_profile["index"]
        owner_name = owner_profile.get("name") or "主人%d" % owner_index

        self.speak(self.search_each_text % owner_name, wait=True)
        rospy.loginfo("开始寻找第 %d 位主人：%s", owner_index, owner_name)

        # 只认这一位：其它主人从镜头前走过不会被当成本轮目标
        result = self.scan_for_owner_for_target(
            target_owner_index=owner_index,
            not_found_text="",  # 这一轮的话术由本方法统一组织
            # 居中/播报/动作识别由本方法自己做，别让扫描内部再做一遍，
            # 否则同一套动作识别会跑两次（实测一次多花 11 秒）
            followup=False,
        )
        if result is None:
            rospy.logwarn("第 %d 位主人 %s 没有找到", owner_index, owner_name)
            self.speak(self.owner_missing_text % owner_name, wait=True)
            return None
        return self.check_found_owner(result)

    def check_found_owner(self, result):
        """认到人之后的统一处理：居中 → 报姓名 → 动作识别 → 报动作。"""
        owner_index = result.get("owner_index")
        owner_name = result.get("owner_name") or "主人%s" % owner_index
        centered = self.center_owner_in_camera(result)
        if not centered:
            rospy.logwarn("主人居中不稳定，跳过居中，继续动作识别")

        # 认人 + 报姓名（父类那句被抑制，这里合并成一句说）
        self.announce_owner_result(result)
        self.speak(self.owner_action_start_text % owner_name, wait=True)

        # 动作识别期间给一句"正在识别"，避免看起来像卡住
        if str(self.owner_action_wait_text or "").strip():
            self.speak(self.owner_action_wait_text, wait=False)

        action_result = self.run_action_recognition_for_owner(result)
        action = (action_result or {}).get("action", "unknown")
        place = (action_result or {}).get("place", "unknown")
        rospy.loginfo(
            "第 %d 位主人 %s 动作结论：action=%s place=%s",
            owner_index,
            owner_name,
            action,
            place,
        )
        speech, next_step = self.owner_feedback(action, place, owner_name)
        self.speak(speech, wait=True)
        # 播报完，按"下一步动作"分派到对应的功能模块（见下方预留接口）
        self.dispatch_next_step(
            next_step, owner_name, action, owner_bbox=result.get("bbox")
        )
        self.publish_status(
            "owner_checked",
            owner_index=owner_index,
            owner_name=owner_name,
            action=action,
            place=place,
            score=result.get("score"),
        )
        return {
            "owner_index": owner_index,
            "owner_name": owner_name,
            "action": action,
            "place": place,
            "score": result.get("score"),
        }

    def run_any_order_round(self):
        """谁先出现就先认谁，逐房间找。

        返回 (已检查列表, 没找到的姓名列表)。
        """
        return self.run_waypoint_search_round(any_order=True)

    def run_ordered_round(self):
        """按注册顺序一位一位找（~search_any_order:=false），同样逐房间。

        注意这个模式比默认模式慢：每个房间里都要对"还没找到的每一位"各扫一整圈，
        扫不到人的那圈纯属白转（一整圈约 40 秒）。调试用可以，比赛用默认模式。
        """
        return self.run_waypoint_search_round(any_order=False)

    # ------------------------------------------------------------------
    # 逐航点找人（导航真正接进来的地方）
    # ------------------------------------------------------------------
    def search_positions(self):
        """本次要去找的"位置"列表。

        开导航 → 是 ~search_waypoints 里的房间名；不开导航 → 只有一个空串，
        代表"就在原地转圈"（老行为，调试识别算法时用）。
        """
        if self.navigate_enabled and self.search_waypoints:
            return list(self.search_waypoints)
        return [""]

    def announce_and_navigate_to(self, waypoint_name):
        """播报"正在前往X"，然后导航过去。返回 True=已经就位。"""
        label = self.waypoint_speech_name(waypoint_name)
        speech = ""
        if str(self.search_go_text or "").strip():
            speech = self.search_go_text % label
        # 把"哪个航点 / 中文名 / 实际播报的那句话"打在**同一行**上。
        # 以前分成两行（`前往第 N 个房间：kitchen（厨房）` 和 `TTS: ...`），
        # 现场听到的和日志里的房间对不上时，没法一眼确定是播报文本错了
        # 还是音频那一层出的问题 —— 这一行就是为了消掉这个歧义。
        rospy.loginfo(
            "前往 %s（%s）→ 播报「%s」",
            waypoint_name,
            label,
            speech or "（不播报）",
        )
        if speech:
            self.speak(speech, wait=True)
        arrived = self.navigate_to_search_position(waypoint_name)
        if not arrived:
            rospy.logwarn(
                "没能到 %s（%s），就在当前位置找主人", waypoint_name, label
            )
        return arrived

    def room_scan_budget(self):
        """这一间房最多能扫多少秒：不超过 ~scan_timeout，也不超出比赛预算。"""
        budget = self.scan_timeout
        remaining = self.competition_time_remaining()
        if remaining != float("inf"):
            budget = min(
                budget, max(0.0, remaining - self.competition_exit_reserve_seconds)
            )
        return budget

    def handle_owner_inline(self, result, remaining, checked):
        """扫描途中确认到一位主人：就地居中 → 报姓名 → 测动作 → 分派下一步。

        返回 True 表示"请结束这一圈的扫描"（时间预算已经用完）。
        """
        owner_index = result.get("owner_index")
        if owner_index in remaining:
            remaining.remove(owner_index)
        record = self.check_found_owner(result)
        if record is not None:
            checked.append(record)
        return self.competition_should_wrap_up()

    def scan_room_in_one_round(self, remaining, names, checked):
        """在这一间房转**一圈**，把在场的主人都认出来并当场处理。

        为什么只转一圈就够：主人被确认的那一刻他就在画面中间，所以居中、报姓名、
        动作识别当场就能做完；做完接着转，转到下一位主人时再确认他。
        "认到一位就收工、下一间再回来找另一位"要多转一整圈（约 40 秒），纯浪费。
        实现靠 scan_for_owner_for_target(on_match=...) —— 已确认的那位会从
        "还要认的人"里去掉，不会被重复确认。
        """
        rospy.loginfo(
            "在这一间转一圈找人：还要认 %d 位（%s），这一圈最多 %.0f 秒",
            len(remaining),
            "、".join(names[i] for i in remaining),
            self.scan_timeout,
        )
        # 进房间开始转之前先确认相机活着：掉流时转一圈是纯浪费（识别全空），
        # 而且日志会显得像"这间房没人"。看门狗的恢复梯子会在这段时间里把画面救回来。
        self.hold_until_camera_recovers("进房间找人")
        found_before_scan = len(checked)

        def handle_match(result):
            """就地处理一位主人；按赛制"每间至多一位"立刻结束这一圈。"""
            # handle_owner_inline 是**阻塞**的：居中 → 报姓名 → 动作识别 → 播报
            # → 分派交互（走近 / 伸臂协助 / 问需要什么帮助），全部做完才返回。
            # 所以下面这两句一定发生在"动作识别和交互都结束之后"。
            wrap_up = self.handle_owner_inline(result, remaining, checked)
            if self.one_owner_per_room:
                rospy.loginfo(
                    "本间已找到 1 位主人（赛制：每间至多一位）→ 结束本间扫描，去下一个航点"
                )
                return True   # 真值 = 请 scan_for_owner_for_target 立刻收圈
            return bool(wrap_up)

        self.scan_for_owner_for_target(
            allowed_owner_indices=set(remaining),
            not_found_text="",       # 话术由总任务统一组织
            followup=False,          # 居中/播报/动作由 handle_owner_inline 做
            on_match=handle_match,
        )
        if self.one_owner_per_room and len(checked) > found_before_scan:
            pass   # 已经在 handle_match 里打过日志了
        elif remaining:
            rospy.loginfo(
                "这一间转完了，还剩 %d 位没找到（%s）",
                len(remaining),
                "、".join(names[i] for i in remaining),
            )
        else:
            rospy.loginfo("这一圈把剩下的人都认齐了")

    def scan_ordered_at_current_position(self, remaining, names, profiles_by_index):
        """在当前站立位置按注册顺序找人。返回这一处认到的记录列表。"""
        checked = []
        for owner_index in list(remaining):
            if rospy.is_shutdown() or self.competition_should_wrap_up():
                break
            record = self.find_and_check_owner(profiles_by_index[owner_index])
            if record is None:
                continue
            remaining.remove(owner_index)
            checked.append(record)
            if self.one_owner_per_room:
                # 赛制：每个房间至多一位主人 → 认到一位就结束这里
                rospy.loginfo(
                    "本处已找到 1 位主人（赛制：每间至多一位）→ 结束此处，去下一个航点"
                )
                break
        return checked

    def run_waypoint_search_round(self, any_order=True):
        """逐房间找人 → 每位主人居中 → 报姓名 → 检测动作。

        返回 (已检查列表, 没找到的姓名列表)。
        """
        names = {
            profile["index"]: (profile.get("name") or "主人%d" % profile["index"])
            for profile in self.owner_profiles
        }
        profiles_by_index = {
            profile["index"]: profile for profile in self.owner_profiles
        }
        remaining = [profile["index"] for profile in self.owner_profiles]
        checked = []
        # 每个房间"这一间找到了几位主人"（备用航点的跳过判断要用）
        room_found = {}

        for position_index, position in enumerate(self.search_positions()):
            if not remaining or rospy.is_shutdown():
                break
            if self.competition_should_wrap_up():
                rospy.logwarn(
                    "时间预算用尽（剩 %.0f 秒），不再寻找剩下的 %s",
                    self.competition_time_remaining(),
                    "、".join(names[i] for i in remaining),
                )
                break
            # ★ 备用航点判断（放在最前面：要跳过的房间不参与时间/预算计算）
            fallback_anchor = self.fallback_waypoint_anchors.get(str(position or ""))
            if fallback_anchor:
                found_at_anchor = room_found.get(fallback_anchor)
                if found_at_anchor is None:
                    rospy.logwarn(
                        "备用航点 %s：主航点 %s 这一轮还没扫过，还是去 %s 看一眼",
                        position,
                        fallback_anchor,
                        position,
                    )
                elif found_at_anchor > 0:
                    rospy.loginfo(
                        "在 %s 找到了 %d 位主人 → 跳过备用航点 %s（%s）",
                        fallback_anchor,
                        found_at_anchor,
                        position,
                        self.waypoint_speech_name(position),
                    )
                    continue
                else:
                    rospy.loginfo(
                        "在 %s 一位都没找到 → 去备用航点 %s（%s）",
                        fallback_anchor,
                        position,
                        self.waypoint_speech_name(position),
                    )
            if position:
                # 出发前先算一笔账：走一趟 + 到了再转一圈，至少要这么多秒。
                # 不够就别去 —— 去了也扫不了，只会白跑一趟、还挤掉回出口的时间。
                headroom = (
                    self.competition_time_remaining()
                    - self.competition_exit_reserve_seconds
                )
                needed = self.min_navigation_seconds + self.min_room_scan_seconds
                if headroom < needed:
                    rospy.logwarn(
                        "去 %s 走一趟再转一圈至少要 %.0f 秒，剩余只有 %.0f 秒，"
                        "这间就不去了，准备收尾",
                        position,
                        needed,
                        headroom,
                    )
                    break
                rospy.loginfo(
                    "前往第 %d 个房间：%s（%s）",
                    position_index + 1,
                    position,
                    self.waypoint_speech_name(position),
                )
                # 每个新房间重新开始计"连续失败"：这一间失败到上限只放弃这一间，
                # 下一间照常导航（2026-10-16 现场要求）。
                self.reset_room_navigation_failures(position)
                self.announce_and_navigate_to(position)
                self.log_time_budget("到达 %s" % position)
            # 记住"当前在哪个房间"：动作播报要用（主人张三正在客厅坐着）
            self.current_room_waypoint = str(position or "")
            # 一间房至少要留 ~min_room_scan_seconds 秒才值得进去扫：
            # 一圈约 40 秒，剩下的时间连一圈都转不完，进去就是白转，
            # 不如把时间留给"走回出口"。
            budget = self.room_scan_budget()
            if budget < self.min_room_scan_seconds:
                rospy.logwarn(
                    "剩余预算只够 %.0f 秒扫这一间（< %.0f 秒），不再找了，准备收尾",
                    budget,
                    self.min_room_scan_seconds,
                )
                break
            # 把"这一圈"的预算压进 scan_timeout：扫描内部的超时上限就是它，
            # 扫描里就地处理主人花掉的时间会被自动补回（见 on_match 处理）。
            saved_scan_timeout = self.scan_timeout
            self.scan_timeout = budget
            # 这一间扫完能新增几位主人？（备用航点靠这个数字决定去不去）
            found_before = len(checked)
            try:
                if any_order:
                    self.scan_room_in_one_round(remaining, names, checked)
                else:
                    checked.extend(
                        self.scan_ordered_at_current_position(
                            remaining, names, profiles_by_index
                        )
                    )
            finally:
                self.scan_timeout = saved_scan_timeout
            room_found[str(position or "")] = len(checked) - found_before
            rospy.loginfo(
                "本间小结：%s（%s）找到 %d 位；累计 %d 位，还剩 %s",
                position or "原地",
                self.waypoint_speech_name(position) if position else "原地",
                room_found[str(position or "")],
                len(checked),
                "、".join(names[i] for i in remaining) or "（无）",
            )

            # 这个房间的流程走完了：看一眼顺手累积的垃圾候选，有就播报。
            # 播报是零额外耗时的 —— 累积是在 detections_callback 里做的，
            # 这里只是把已经攒够的证据说出来（规则 2.6-6 以语音播报为判分依据）。
            self.announce_trash_if_found(position or "原地")

        missing = []
        for index in remaining:
            self.speak(self.owner_missing_text % names[index], wait=True)
            missing.append(names[index])
        return checked, missing

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self):
        # 计时从这里开始：注册 + 找人 + 动作识别 + 回出口，全在 8 分钟里。
        self.competition_started_time = time.time()
        # 先把本次运行的关键开关打出来。
        # 为什么必须打：命令行里的 `_reuse_existing_profile:=true` 之类会以
        # **私有参数**的形式留在 ROS master 上，节点退出也不会清；下一次不传
        # 这个参数的运行会读到上次残留的值（实测：跑完 ③g 紧接着跑 ③f，
        # ③f 会莫名其妙跳过注册直接去找人）。打出来一眼就能发现。
        rospy.loginfo(
            "本次运行参数：注册人数=%d 复用已有档案=%s 导航=%s 找人顺序=%s "
            "比赛时长=%.0fs 回出口预留=%.0fs",
            self.owner_count,
            "是（跳过注册！）" if self.reuse_existing_profile else "否（重新注册）",
            "开" if self.navigate_enabled else "关",
            "谁先出现先认谁" if self.search_any_order else "按注册顺序",
            self.competition_total_seconds,
            self.competition_exit_reserve_seconds,
        )
        if self.navigate_enabled:
            rospy.loginfo(
                "导航已启用：房间顺序=%s，回出口航点=%s，单航点超时=%.0fs，"
                "同一目标（每间房/起点/回出口）连续失败 %d 次就只放弃该目标，"
                "下一间重新计数继续导航",
                "、".join(
                    "%s(%s)" % (name, self.waypoint_speech_name(name))
                    for name in self.search_waypoints
                )
                or "（空！只会原地找）",
                self.exit_waypoint_name or "（无）",
                self.waypoint_goal_timeout,
                self.navigation_failure_limit,
            )
        # ---- 第 0 步：等输入就绪、把模型加载好 ----
        if not self.check_hardware():
            rospy.logerr(
                "硬件没就绪：本次任务直接结束（不导航、不开车）。"
                "修好硬件后重新运行即可。"
            )
            return
        self.wait_for_tts()
        if not self.wait_for_asr():
            return
        self.wait_for_camera_inputs()
        self.start_camera_watchdog()
        self.init_yolo_window()
        self.init_reid_backend()
        self.init_face_recognizer()
        # 姿态模型现在同时兼任"人体检测器"（YOLO-World 已停用），
        # 注册阶段采人脸/身形都要靠它给候选框，必须先加载。
        self.init_pose_helper()
        # 注册这两分钟完全不需要看垃圾，把垃圾检测节点先按住（省 CPU 给注册用）
        self.set_trash_detector_paused(True, "注册阶段不需要看垃圾")

        # ---- 第 0.5 步：先去指定航点（例如客厅），在那里注册（2026-10-15 新流程）----
        # 现场赛程是"机器人从门口进客厅 → 在客厅注册主人 → 再逐房间找人"。
        # ~start_waypoint 为空 = 保持老流程（在起点原地注册）。
        # 出发前先把 AMCL 的初始位姿设好（替代每次手点 2D Pose Estimate）
        if self.auto_initial_pose:
            self.publish_auto_initial_pose()
        else:
            # 关掉自动初值 = 完全靠人在 RViz 里点。开局把当前位姿打进日志，
            # 点没点 / 点歪了，一眼能看出来（否则就是一串 Failed to get a plan.）。
            self.report_manual_initial_pose()
        if self.start_waypoint:
            speech_name = self.waypoint_speech_name(self.start_waypoint)
            self.speak(
                self.start_waypoint_text % speech_name,
                wait=True,
            )
            # 起点也是"新目标"：失败计数从 0 开始
            self.reset_room_navigation_failures(self.start_waypoint)
            arrived = self.navigate_to_waypoint(self.start_waypoint)
            self.log_time_budget("到达注册点 %s" % self.start_waypoint)
            if not arrived:
                rospy.logwarn(
                    "没能导航到注册点 %s，就地注册（后面流程照常）", self.start_waypoint
                )
            # 注册点的房间名先记下来：万一紧接着就在这里做动作播报，也能带上房间
            self.current_room_waypoint = str(self.start_waypoint)

        # ---- 第 1 步：注册 ----
        if self.reuse_existing_profile and self.load_all_owner_profiles():
            rospy.loginfo("Loaded existing owner profiles from %s", self.profile_dir)
        else:
            self.record_all_owners()
        if not self.owner_profiles:
            raise RuntimeError("owner profiles are not ready")
        rospy.loginfo(
            "注册完成：%s",
            ", ".join(
                "%d=%s" % (profile["index"], profile.get("name") or "")
                for profile in self.owner_profiles
            ),
        )
        self.log_time_budget("注册完成")

        # ---- 注册完成后：请志愿者就位，原地停 N 秒（不动），再开始寻找 ----
        # 为什么要停：志愿者听到这句话才走到位/躺好，机器人立刻开找会白扫一圈。
        # 期间**不发任何速度指令**（stop_base），只等时间。
        if str(self.search_ready_text or "").strip():
            self.speak(self.search_ready_text, wait=True)
        if self.search_ready_pause > 0.0:
            rospy.loginfo(
                "请志愿者就位：原地停 %.1f 秒（不动）后再开始寻找",
                self.search_ready_pause,
            )
            self.stop_base()
            rospy.sleep(self.search_ready_pause)
            self.log_time_budget("志愿者就位")

        # ---- 第 2 步：逐房间找、逐个检测 ----
        # 从现在开始相机要扫房间了，垃圾就在这时候看
        self.set_trash_detector_paused(False, "开始逐房间寻找主人")
        self.speak(self.search_start_text, wait=True)
        if self.search_any_order:
            # 谁先出现先认谁：到一个房间先把这间里还没找到的人都认出来，
            # 这间没人了再换下一间；三位都找到就提前收尾（剩下的房间不去）。
            checked, missing = self.run_any_order_round()
        else:
            # 按注册顺序一位一位找（慢，调试用）
            checked, missing = self.run_ordered_round()
        self.log_time_budget("寻找主人结束")
        # 收尾前再看一眼垃圾：有些候选是在最后一段路上才攒够的
        self.announce_trash_if_found("收尾")

        # ---- 第 3 步：收尾 ----
        if rospy.is_shutdown():
            return
        # 复用模式下动作节点是常驻的，任务结束统一关掉，别留在后台
        try:
            self.shutdown_action_recognition()
        except Exception as exc:
            rospy.logwarn("关闭动作识别节点时出错：%s", exc)
        self.speak(self.all_done_text, wait=True)
        self.set_trash_detector_paused(True, "寻找主人已结束")
        # 收尾：全部主人检查完之后导航回出口（自主离场）。
        # 不导航模式下它只打一句日志，不会动。
        self.return_to_exit()
        rospy.loginfo(
            "总任务结束：找到并检测 %d 位（%s）；没找到 %d 位（%s）",
            len(checked),
            "、".join("%s=%s" % (item["owner_name"], item["action"]) for item in checked)
            or "无",
            len(missing),
            "、".join(missing) or "无",
        )
        self.publish_status(
            "all_owners_finished",
            checked=checked,
            missing=missing,
        )


def main():
    rospy.init_node("task_all_owners")
    node = AllOwnersTask()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("all-owners task failed: %s", exc)
        # 相机断流时给一句准确的提示，免得把"没画面"误当成"没识别到"
        if getattr(node, "camera_lost", False):
            try:
                node.speak("摄像头没有画面了，请检查摄像头连接后重试。", wait=True)
            except Exception:
                pass
        raise
    finally:
        try:
            node.stop_camera_watchdog()
        except Exception:
            pass
        node.stop_base()


if __name__ == "__main__":
    main()
