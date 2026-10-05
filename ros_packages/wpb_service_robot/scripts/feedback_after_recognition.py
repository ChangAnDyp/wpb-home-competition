#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动作 / 语音识别之后的反馈模块。

设计成"一个文件两种用法"，与动作识别模块（圈2）保持一致：

  1) 被主任务 import（不带 ROS 通信，只是普通函数调用）
        fb = core.feedback_for_action("waving", owner_name="")
        fb.speech     -> "主人正在挥手"
        fb.next_step  -> "approach_and_ask_help"

  2) 单独运行（分模块测试）
        启动本文件后订阅
          /pose_action_recognition/result  （动作识别结果）
          /voice/asr_text                  （语音识别结果）
        收到结果就调用上面的纯逻辑，并把播报发到 /voice/say

依赖的比赛规则（《最新赛事规则.pdf》2.6 评分细则）：
  * 识别主人、判定行为，都以"语音播报"为主要判分依据 —— 本模块是分数兑现层
  * 播报格式统一为「主人{姓名}正在{动作}」
  * 姓名还没接上（脸部注册未完成）时省略姓名 → "主人正在挥手"
  * 暂不区分床 / 沙发 / 椅子

本模块只负责"说什么 + 下一步做什么"（next_step 只是个名字），
不去真的开机械臂或走导航 —— 那些由后续模块实现，主任务按 next_step 分派。
"""

import json

import rospy
from std_msgs.msg import String


DEFAULT_RESULT_TOPIC = "/pose_action_recognition/result"
DEFAULT_ASR_TOPIC = "/voice/asr_text"
DEFAULT_SAY_TOPIC = "/voice/say"

# 主人姓名占位。脸部注册模块完成后，由主任务通过 owner_name 传进来。
DEFAULT_OWNER_NAME = ""

# 动作 -> 播报短语（不含"主人"和姓名前缀）
# 注意：刻意不体现床/沙发/椅子（按当前决定，暂不区分家具）
ACTION_PHRASE = {
    "sitting": "正在坐着",
    "lying": "正在躺着",
    "waving": "正在挥手",
    "fallen": "摔倒了",
    "sudden_fall": "突然摔倒了",
}

# 动作 -> 下一步该做什么（只给名字，具体执行由后续模块实现）
ACTION_NEXT_STEP = {
    "sitting": "ask_switch",
    "lying": "ask_switch",
    "fallen": "approach_and_assist",
    "sudden_fall": "approach_and_assist",
    "waving": "approach_and_ask_help",
}

UNKNOWN_SPEECH = "暂时无法判断主人的动作。"
UNKNOWN_NEXT_STEP = "retry"

# 语音意图 -> 播报。框架已搭好，等语音意图识别接入后直接往这里填。
VOICE_INTENT_SPEECH = {
    "switch_on": "好的，我这就去把开关打开。",
    "switch_off": "好的，我这就去把开关关掉。",
}
VOICE_INTENT_NEXT_STEP = {
    "switch_on": "operate_switch",
    "switch_off": "operate_switch",
}


class Feedback(object):
    """一次反馈的结果：说什么（speech）+ 接下来做什么（next_step）。"""

    __slots__ = ("speech", "next_step", "source", "detail")

    def __init__(self, speech, next_step, source, detail=None):
        self.speech = speech
        self.next_step = next_step
        self.source = source
        self.detail = detail or {}

    def as_dict(self):
        return {
            "speech": self.speech,
            "next_step": self.next_step,
            "source": self.source,
            "detail": self.detail,
        }

    def __repr__(self):
        return "Feedback(speech=%r, next_step=%r, source=%r)" % (
            self.speech,
            self.next_step,
            self.source,
        )


def compose_speech(phrase, owner_name=DEFAULT_OWNER_NAME):
    """拼出「主人{姓名}{短语}」。

    owner_name 为空  -> "主人正在挥手"
    owner_name 有值  -> "主人张三正在挥手"
    """
    name = (owner_name or "").strip()
    return "主人%s%s" % (name, phrase)


def feedback_for_action(action, place="unknown", owner_name=DEFAULT_OWNER_NAME):
    """动作识别结果 -> 反馈。纯逻辑，不依赖 ROS。"""
    key = str(action or "").strip().lower()
    phrase = ACTION_PHRASE.get(key)
    if phrase is None:
        return Feedback(
            speech=UNKNOWN_SPEECH,
            next_step=UNKNOWN_NEXT_STEP,
            source="pose",
            detail={"action": key, "place": place, "known": False},
        )
    return Feedback(
        speech=compose_speech(phrase, owner_name),
        next_step=ACTION_NEXT_STEP[key],
        source="pose",
        detail={"action": key, "place": place, "known": True},
    )


def feedback_for_voice(transcript, intent=None, owner_name=DEFAULT_OWNER_NAME):
    """语音识别结果 -> 反馈。

    框架已搭好：intent 为空或还没支持的意图时返回 None（不产生反馈）。
    等语音意图识别接进来，把意图名填进 VOICE_INTENT_SPEECH 即可生效。
    """
    key = str(intent or "").strip().lower()
    speech = VOICE_INTENT_SPEECH.get(key)
    if speech is None:
        return None
    return Feedback(
        speech=speech,
        next_step=VOICE_INTENT_NEXT_STEP.get(key, "unknown"),
        source="voice",
        detail={"transcript": transcript, "intent": key},
    )


class FeedbackAfterRecognitionNode(object):
    """单独运行时的 ROS 外壳：订阅识别结果，调用上面的纯逻辑并播报。"""

    def __init__(self):
        self.result_topic = rospy.get_param("~result_topic", DEFAULT_RESULT_TOPIC)
        self.asr_topic = rospy.get_param("~asr_topic", DEFAULT_ASR_TOPIC)
        self.say_topic = rospy.get_param("~say_topic", DEFAULT_SAY_TOPIC)
        self.owner_name = rospy.get_param("~owner_name", DEFAULT_OWNER_NAME)
        self.speak_enabled = bool(rospy.get_param("~speak_enabled", True))
        self.voice_feedback_enabled = bool(
            rospy.get_param("~voice_feedback_enabled", True)
        )

        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=10)
        self.action_sub = rospy.Subscriber(
            self.result_topic,
            String,
            self.action_result_callback,
            queue_size=5,
        )
        self.asr_sub = None
        if self.voice_feedback_enabled:
            self.asr_sub = rospy.Subscriber(
                self.asr_topic,
                String,
                self.asr_callback,
                queue_size=5,
            )
        rospy.loginfo(
            "Feedback node ready: result=%s asr=%s say=%s owner_name=%r",
            self.result_topic,
            self.asr_topic if self.asr_sub is not None else "(disabled)",
            self.say_topic,
            self.owner_name,
        )

    def action_result_callback(self, message):
        try:
            data = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(data, dict) or data.get("error"):
            return
        feedback = feedback_for_action(
            data.get("action"),
            data.get("place", "unknown"),
            self.owner_name,
        )
        self.publish_feedback(feedback)

    def asr_callback(self, message):
        # 语音意图识别还没接进来，这里先固定传 intent=None（不产生反馈）。
        # 接入后把识别出的意图传进来即可。
        feedback = feedback_for_voice(
            message.data,
            intent=None,
            owner_name=self.owner_name,
        )
        self.publish_feedback(feedback)

    def publish_feedback(self, feedback):
        if feedback is None:
            return
        rospy.loginfo(
            "Feedback: source=%s next_step=%s speech=%s",
            feedback.source,
            feedback.next_step,
            feedback.speech,
        )
        if self.speak_enabled:
            self.say_pub.publish(String(data=feedback.speech))


def main():
    rospy.init_node("feedback_after_recognition")
    FeedbackAfterRecognitionNode()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
