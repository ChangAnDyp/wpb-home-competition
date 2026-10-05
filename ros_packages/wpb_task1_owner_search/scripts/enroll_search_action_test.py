#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主人注册 → 旋转寻找 → 识别主人 → 检测动作 —— 全流程测试脚本。

流程
----
  第 1 步  主人注册（复用 face_body_enroll_test：姓名 → 确认 → 人脸 → 身形）
           第 2 位主人可以语音说"跳过"跳过（第 1 位不能跳）
  第 2 步  播报"开始寻找主人。"，然后原地旋转寻找
  第 3 步  找到主人后播报"找到主人XXX了，现在开始检测您的动作。"
           （找到与开始检测合并成一句话）
  第 4 步  调动作识别：只分析当前主人的 ROI，只检测一次
  第 5 步  动作识别节点播报检测到的动作，然后本脚本结束

实现方式
--------
继承 face_body_enroll_test.FaceBodyEnrollTest（后者又继承
owner_voice_reid_test.OwnerVoiceReidTest），把注册、寻找、身份识别、
动作识别全部复用，只补两句播报、并把"找到主人 + 开始检测动作"合并。
本文件不修改任何已有文件。

常用参数
--------
  ~owner_count               注册几位主人（两人测试填 2）
  ~search_start_text         开始寻找时的播报
  ~action_start_text         找到主人后的合并播报（%s = 主人姓名）
  ~search_failed_text        没找到主人时的播报
  ~action_timeout            等动作识别结果的超时（秒）
"""

import os
import sys
import math

import rospy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from face_body_enroll_test import FaceBodyEnrollTest  # noqa: E402


class EnrollSearchActionTest(FaceBodyEnrollTest):
    """注册 -> 寻找 -> 识别 -> 动作，找到一位主人并检测完动作就结束。"""

    def __init__(self):
        super().__init__()
        # 默认扫满一圈 360°，而不是父类的 180°：
        # 人站在机器人背后时只扫半圈永远找不到他（躺着/摔倒的人尤其容易
        # 落在身后视野外）。可用 _scan_total_angle 覆盖。
        self.scan_total_angle = max(
            0.1, float(rospy.get_param("~scan_total_angle", math.radians(400.0)))
        )
        # 扫 400°（比一圈多 40°，给起止接缝留重叠），约需 41 秒，
        # 超时放宽到 90 秒以免"刚好扫不完"。
        self.scan_timeout = max(
            1.0, float(rospy.get_param("~scan_timeout", 90.0))
        )
        self.search_start_text = rospy.get_param("~search_start_text", "开始寻找主人。")
        self.search_failed_text = rospy.get_param("~search_failed_text", "没有找到主人。")
        # %s 会被替换成主人的姓名，例如："找到主人张三了，现在开始检测您的动作。"
        self.action_start_text = rospy.get_param(
            "~action_start_text", "找到主人%s了，现在开始检测您的动作。"
        )
        # 父类的 announce_owner_result 会单独播报"识别到主人X"；
        # 本脚本把它抑制掉，改由下面合并成一句话播报。
        self._suppress_owner_found_speech = False

    def speak(self, text, wait=True):
        """在抑制窗口内不发声，只记日志（其余行为不变）。"""
        if self._suppress_owner_found_speech:
            rospy.loginfo("TTS（已抑制，稍后合并播报）: %s", text)
            return
        return super().speak(text, wait=wait)

    def announce_owner_result(self, result):
        """父类在这里播报"识别到主人X"；改成统一放到开始检测动作那句里说。"""
        self._suppress_owner_found_speech = True
        try:
            return super().announce_owner_result(result)
        finally:
            self._suppress_owner_found_speech = False

    def run_owner_action_recognition(self, owner_result):
        """开始调动作识别之前，播报合并后的那一句。"""
        owner_name = ""
        if owner_result:
            owner_name = owner_result.get("owner_name") or ""
        if not owner_name:
            owner_name = self.owner_name or "主人"
        self.speak(self.action_start_text % owner_name, wait=True)
        return super().run_owner_action_recognition(owner_result)

    def run(self):
        # 关键开关先打出来：命令行里的 `_reuse_existing_profile:=true` 会以
        # 私有参数形式留在 ROS master 上，节点退出也不清；下次不传这个参数
        # 的运行会读到上次残留的值，表现为"莫名其妙跳过注册"。
        rospy.loginfo(
            "本次运行参数：注册人数=%d 复用已有档案=%s",
            self.owner_count,
            "是（跳过注册！）" if self.reuse_existing_profile else "否（重新注册）",
        )
        # ---- 前半段与父类一致：等语音/相机 -> 初始化 -> 注册 ----
        self.wait_for_tts()
        if not self.wait_for_asr():
            return
        self.wait_for_camera_inputs()
        self.init_yolo_window()
        self.init_reid_backend()
        self.init_face_recognizer()
        # 姿态模型必须在注册之前加载：它现在同时充当"人体检测器"，
        # 注册时采人脸/身形都要靠它给候选框（YOLO-World 已停用，
        # 不提前加载的话注册阶段一个候选都拿不到）。
        self.init_pose_helper()
        if self.reuse_existing_profile and self.load_all_owner_profiles():
            rospy.loginfo(
                "Loaded existing owner profiles from %s", self.profile_dir
            )
        else:
            self.record_all_owners()
        if not self.owner_profiles:
            if self.owner_embedding is not None:
                self.remember_current_owner_profile()
            else:
                raise RuntimeError("owner profiles are not ready")

        rospy.loginfo(
            "注册完成：%s",
            ", ".join(
                "%d=%s" % (profile["index"], profile["name"])
                for profile in self.owner_profiles
            ),
        )

        # ---- 第 2 步：播报并开始旋转寻找 ----
        self.speak(self.search_start_text, wait=True)
        rospy.loginfo("开始原地旋转寻找主人")
        result = self.scan_for_owner()

        # ---- 第 3/4 步由 scan_for_owner 内部触发：
        #      找到 -> announce_owner_result(不发声) -> run_owner_action_recognition
        #      （后者先播报合并句，再调动作识别节点，只分析主人 ROI、只测一次）
        if result is None:
            rospy.logwarn("没有找到主人，流程结束")
            self.speak(self.search_failed_text, wait=True)
            return

        rospy.loginfo(
            "全流程结束：找到 %s（index=%s）并完成一次动作检测",
            result.get("owner_name", "unknown"),
            result.get("owner_index"),
        )


def main():
    rospy.init_node("enroll_search_action_test")
    node = EnrollSearchActionTest()
    try:
        node.run()
    except Exception as exc:
        rospy.logerr("enroll/search/action test failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
