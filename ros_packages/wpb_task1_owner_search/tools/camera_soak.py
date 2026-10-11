#!/usr/bin/env python3
# coding: utf-8
"""彩色流"泡机"监视器：不需要任务，只看相机自己稳不稳。

用途（找根因的关键实验）：
    python3 tools/camera_soak.py --minutes 10
      ① 只起栈（⑦/⑦b，别跑任务）泡 10 分钟 —— 掉流 = 设备/主机侧问题
      ② 起栈 + 跑 ⑫ 泡 10 分钟 —— 只在跑任务时掉 = 负载/订阅相关

每 5 秒打印一次帧率；掉流（>2 秒没有新帧）时立刻打印一行，恢复时再打印
一行（含断流时长）。退出时给出总统计。它只是订阅，不动任何东西。
"""

import argparse
import time

import rospy
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--topic", default="/kinect2/qhd/image_color_rect")
    parser.add_argument("--gap", type=float, default=2.0, help="超过这么多秒算掉流")
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("camera_soak", anonymous=True)
    state = {"last": None, "count": 0}

    def cb(_msg):
        state["last"] = time.time()
        state["count"] += 1

    rospy.Subscriber(args.topic, Image, cb, queue_size=1, buff_size=2 ** 24)
    rospy.loginfo(
        "泡机开始：topic=%s 时长=%.1f 分钟 掉流阈值=%.1fs（另外开一个终端跑任务做对照）",
        args.topic,
        args.minutes,
        args.gap,
    )

    deadline = time.time() + args.minutes * 60.0
    drops = []
    drop_start = None
    window_start = time.time()
    window_frames = 0
    while not rospy.is_shutdown() and time.time() < deadline:
        rospy.sleep(0.2)
        now = time.time()
        last = state["last"]
        if last is not None and now - last > args.gap:
            if drop_start is None:
                drop_start = last
                print("[%s] 掉流开始（最近一帧 %.1f 秒前）"
                      % (time.strftime("%H:%M:%S"), now - last))
        elif drop_start is not None:
            drops.append((drop_start, now))
            print("[%s] 画面恢复：断流 %.1f 秒" % (time.strftime("%H:%M:%S"), now - drop_start))
            drop_start = None

        if now - window_start >= 5.0:
            frames = state["count"] - window_frames
            elapsed = now - window_start
            print("[%s] %.1f Hz（窗口内 %d 帧，累计 %d 帧）"
                  % (time.strftime("%H:%M:%S"), frames / max(elapsed, 1e-6),
                     frames, state["count"]))
            window_start = now
            window_frames = state["count"]

    if state["last"] is None:
        print("整个泡机过程一帧都没收到 —— 相机根本没出图（先跑 camera_health.sh）")
    else:
        print("泡机结束：累计 %d 帧，掉流 %d 次，总断流 %.1f 秒"
              % (state["count"], len(drops), sum(e - s for s, e in drops)))


if __name__ == "__main__":
    main()
