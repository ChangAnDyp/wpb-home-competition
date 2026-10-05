#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""总任务：多位主人注册 → 逐位寻找 → 认人报姓名 → 报动作。

流程（对应比赛"识别主人 + 判定行为"那条主线）
------------------------------------------------
  第 0 步  等语音 / 相机 / 模型就绪
  第 1 步  主人注册（默认 3 位：姓名 → 确认 → 人脸 → 身形）
           第 2 位起可以语音说"跳过"；已注册过时可用 ~reuse_existing_profile 跳过
  第 2 步  对每一位主人循环：
             2.1  走到找人的位置      ← 导航占位接口，当前原地不动
             2.2  原地旋转扫描，只认"这一位"主人
             2.3  把主人居中到画面中间
             2.4  播报"找到主人X了，现在开始检测您的动作"
             2.5  调动作识别节点，播报"主人X正在Y"
  第 3 步  汇总播报本次检查结果

设计说明
--------
* 本文件是编排层：注册 / 认人 / 扫描全部复用父类 FaceBodyEnrollTest 的方法，
  动作识别交给独立节点 qwen_action_recognition_node.py（两个进程之间走话题），
  所以这里不重复实现任何算法。
* "主人X正在Y"这句话由 feedback_after_recognition.py 的纯逻辑拼出来。
  为了避免播两遍，动作识别节点自己的那句要关掉（~action_speak_enabled，
  本文件默认关；老流程 ③d 默认仍是开，行为不变）。
* 导航目前是空实现（~navigate_enabled 默认 false，即原地转圈找人）。
  以后有地图了，把 navigate_to_search_position() 里的 TODO 换成真正的导航调用
  即可，其余流程不用动。

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
"""

import importlib.util
import math
import os
import sys
import threading
import time

import rospy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from face_body_enroll_test import FaceBodyEnrollTest  # noqa: E402


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

        # 导航开关沿用父类已有的 ~navigate_enabled，只把默认值改成 false：
        # 现在没有地图，默认原地转圈找人。将来有地图了，命令行加
        # _navigate_enabled:=true 即可切过去（导航方法父类里已经有了）。
        self.navigate_enabled = bool(
            rospy.get_param("~navigate_enabled", False)
        )
        # 动作结论由本文件播报（要带姓名），所以关掉动作节点自己那句
        self.action_speak_enabled = bool(
            rospy.get_param("~action_speak_enabled", False)
        )

        self.search_start_text = rospy.get_param("~search_start_text", "开始寻找主人。")
        self.search_each_text = rospy.get_param("~search_each_text", "正在寻找%s。")
        # "谁先出现先认谁"模式下，第 2 轮起替换掉"正在寻找X"时说的话
        self.search_next_text = rospy.get_param(
            "~search_next_text", "继续寻找其他主人。"
        )
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
                        rospy.logerr(
                            "相机掉线：%s 已连续 %.1f 秒没有新画面"
                            "（进程还活着，是画面流断了，不是识别算法的问题）",
                            self.image_topic,
                            now - stale_since,
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
          sitting / lying  -> ask_switch
          fallen / sudden_fall -> approach_and_assist
          waving           -> approach_and_ask_help
          unknown          -> retry
        """
        try:
            feedback = self.feedback_module().feedback_for_action(
                action, place, owner_name
            )
            return feedback.speech, feedback.next_step
        except Exception as exc:
            rospy.logwarn("拼动作播报失败，退回简单说法：%s", exc)
            return (
                "主人%s的动作是%s。" % (owner_name or "", action or "unknown"),
                "unknown",
            )

    # ------------------------------------------------------------------
    # 导航占位接口
    # ------------------------------------------------------------------
    def navigate_to_search_position(self, waypoint_name=""):
        """找主人之前"走到位"的那一步。

        现在没有地图，所以默认什么都不做（原地开始扫描）。
        等有了地图和航点，把下面的 TODO 换成真正的导航即可，例如：

            self.load_waypoint_pose("living_room")      # 父类已有
            self.navigate_to_waypoint("living_room")    # 父类已有

        （task1_find_owner_real.py 里有更完整的导航实现可参考：
          清代价地图、发送 move_base 目标、超时与失败重试等。）
        """
        if not self.navigate_enabled:
            rospy.loginfo("导航未启用：原地开始寻找主人")
            return True
        rospy.logwarn(
            "已请求导航，但本文件的导航接口还没接上地图；"
            "先按原地转圈继续（见 navigate_to_search_position 的 TODO）"
        )
        return False

    # ------------------------------------------------------------------
    # 预留接口：还没实现的功能都在这里占位
    # ------------------------------------------------------------------
    # 说明：这些方法现在只打日志、返回 None，主流程调用它们不会出错。
    # 以后实现时**只改这些方法内部**，主流程（run / check_found_owner）不用动。
    # 分派入口是 dispatch_next_step()，它按 feedback_after_recognition.py 给出的
    # next_step 名字路由，名字是两边约定的契约。
    def check_hardware(self):
        """【预留】开机硬件自检（话题/串口/Kinect 是否就绪）。

        参考 task1_find_owner_real.py 的 wait_for_hardware_inputs()。
        返回 True 表示可以开始任务。
        """
        rospy.loginfo("硬件自检：预留接口，暂未实现（默认视为通过）")
        return True

    def navigate_to_waypoint(self, waypoint_name=""):
        """【预留】导航到指定航点（跨房间找人用）。

        父类 OwnerVoiceReidTest 里已经有 load_waypoint_pose() / navigate_to_waypoint()
        的完整实现，接上就能用；本方法留作"总任务自己的导航入口"。
        """
        rospy.loginfo("导航到航点 %s：预留接口，暂未实现", waypoint_name or "(未指定)")
        return None

    def approach_owner(self, owner_name=""):
        """【预留】走近主人（挥手求助 / 摔倒协助都要先用它到位）。"""
        rospy.loginfo("接近主人 %s：预留接口，暂未实现", owner_name or "")
        return None

    def perform_fall_assist(self, owner_name="", action=""):
        """【预留】摔倒后伸机械臂协助。

        task1_find_owner_real.py 里有 perform_fall_assist_arm_motion() 可参考，
        机械臂话题见 wpb_home 包。
        """
        rospy.loginfo(
            "摔倒协助（主人=%s 动作=%s）：预留接口，暂未实现",
            owner_name or "",
            action or "",
        )
        return None

    def ask_switch_instruction(self, owner_name=""):
        """【预留】问主人要不要操作电器开关。"""
        rospy.loginfo(
            "询问电器开关指令（主人=%s）：预留接口，暂未实现", owner_name or ""
        )
        return None

    def operate_switch(self, owner_name=""):
        """【预留】执行电器开关动作。"""
        rospy.loginfo("执行电器开关（主人=%s）：预留接口，暂未实现", owner_name or "")
        return None

    def return_to_exit(self):
        """【预留】全部完成后导航回出口（自主离场）。"""
        rospy.loginfo("返回出口：预留接口，暂未实现")
        return None

    def dispatch_next_step(self, next_step, owner_name="", action=""):
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
        if next_step == "approach_and_assist":
            self.approach_owner(owner_name)
            return self.perform_fall_assist(owner_name, action)
        if next_step == "approach_and_ask_help":
            return self.approach_owner(owner_name)
        if next_step == "ask_switch":
            return self.ask_switch_instruction(owner_name)
        if next_step == "operate_switch":
            return self.operate_switch(owner_name)
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
        return self.run_owner_action_recognition(owner_result)

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
        self.dispatch_next_step(next_step, owner_name, action)
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

    def find_any_remaining_owner(self, remaining_indices):
        """转一圈，认到"还没找到的那几位"里的任意一位就返回（不做居中/播报/动作）。"""
        return self.scan_for_owner_for_target(
            allowed_owner_indices=set(remaining_indices),
            not_found_text="",       # 话术由调用方统一组织
            followup=False,          # 居中/播报/动作识别由调用方自己做
        )

    def run_any_order_round(self):
        """谁先出现就先认谁：认过的排除，转一圈没再认到人就收尾。

        返回 (已检查列表, 没找到的姓名列表)。
        """
        names = {
            profile["index"]: (profile.get("name") or "主人%d" % profile["index"])
            for profile in self.owner_profiles
        }
        remaining = [profile["index"] for profile in self.owner_profiles]
        checked = []
        round_index = 0
        while remaining and not rospy.is_shutdown():
            round_index += 1
            rospy.loginfo(
                "第 %d 轮寻找：剩下 %d 位没找到（%s）",
                round_index,
                len(remaining),
                "、".join(names[i] for i in remaining),
            )
            if round_index > 1 and str(self.search_next_text or "").strip():
                self.speak(self.search_next_text, wait=True)
            result = self.find_any_remaining_owner(remaining)
            if result is None:
                rospy.logwarn("转完一圈没有再认出剩下的人，其余按未找到处理")
                break
            owner_index = result.get("owner_index")
            if owner_index in remaining:
                remaining.remove(owner_index)
            record = self.check_found_owner(result)
            if record is not None:
                checked.append(record)
        missing = []
        for index in remaining:
            self.speak(self.owner_missing_text % names[index], wait=True)
            missing.append(names[index])
        return checked, missing

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self):
        # 先把本次运行的关键开关打出来。
        # 为什么必须打：命令行里的 `_reuse_existing_profile:=true` 之类会以
        # **私有参数**的形式留在 ROS master 上，节点退出也不会清；下一次不传
        # 这个参数的运行会读到上次残留的值（实测：跑完 ③g 紧接着跑 ③f，
        # ③f 会莫名其妙跳过注册直接去找人）。打出来一眼就能发现。
        rospy.loginfo(
            "本次运行参数：注册人数=%d 复用已有档案=%s 导航=%s 找人顺序=%s",
            self.owner_count,
            "是（跳过注册！）" if self.reuse_existing_profile else "否（重新注册）",
            "开" if self.navigate_enabled else "关",
            "谁先出现先认谁" if self.search_any_order else "按注册顺序",
        )
        # ---- 第 0 步：等输入就绪、把模型加载好 ----
        self.check_hardware()
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

        # ---- 第 2 步：逐个找、逐个检测 ----
        self.speak(self.search_start_text, wait=True)
        self.navigate_to_search_position(self.waypoint_name)
        checked = []
        missing = []
        if self.search_any_order:
            # 谁先出现先认谁；认过的这一轮不再找；转一圈没人了就收尾
            checked, missing = self.run_any_order_round()
        else:
            for profile in self.owner_profiles:
                if rospy.is_shutdown():
                    break
                record = self.find_and_check_owner(profile)
                if record is None:
                    missing.append(profile.get("name") or "主人%d" % profile["index"])
                else:
                    checked.append(record)

        # ---- 第 3 步：收尾 ----
        if rospy.is_shutdown():
            return
        # 复用模式下动作节点是常驻的，任务结束统一关掉，别留在后台
        try:
            self.shutdown_action_recognition()
        except Exception as exc:
            rospy.logwarn("关闭动作识别节点时出错：%s", exc)
        self.speak(self.all_done_text, wait=True)
        # 【预留】收尾：全部主人检查完之后回出口（自主离场）
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
