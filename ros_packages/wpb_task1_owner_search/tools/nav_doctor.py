#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导航体检：定位「规划不出路径」到底是谁的锅（只读，不发表任何运动目标）。

为什么需要它：move_base 只会告诉你 Failed to get a plan. / ABORTED，
但同一条失败日志背后可能是三件完全不同的事：

  A. 定位偏了           —— 激光打到的墙和静态地图的墙对不上，实时障碍层
                          于是把「墙」标到了地图的空地上，通道被假墙封死；
  B. 被实时障碍围住     —— 人/家具/门框在膨胀 0.5 m 后形成 1.5 m 禁行带，
                          机器人（或目标点）被整个包住；
  C. 目标点本身就不可达 —— 航点离墙太近（< 0.25 m 就落进致命格），
                          或者中间确实被静态地图的墙隔开。

本脚本把这三件事分开量给你看：

  1. 当前位姿（TF map->base_footprint）与 /amcl_pose 是否一致；
  2. /scan 最近距离、最近若干点（看是不是贴着人/墙）；
  3. 关键：激光端点 vs 静态地图墙。每个激光点投到 map 系后，量它离最近的
     黑格墙有多远：定位准 -> 几乎都贴着墙（0~0.1 m）；定位偏 -> 飘在空地中间。
  4. 全局代价地图里：机器人周围 0.5 m 有多少致命/内切格（被围住？）；
     每个航点格的代价（>=253 就是规划不进去）。
  5. 用当前位姿对每个航点调 /move_base/make_plan，看能不能出路径。

用法（起栈之后，另开一个终端）：

    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    python3 ~/catkin_ws/src/wpb_task1_owner_search/tools/nav_doctor.py

    # 只看某一个航点：
    python3 .../nav_doctor.py --goal living_room

退出码：0 = 没查出问题；1 = 至少有一项不正常（具体看输出结论）。
"""

import argparse
import math
import sys

import numpy as np

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import LaserScan


DEFAULT_WAYPOINTS = "/home/dyp/waypoints.xml"
COSTMAP_TOPIC = "/move_base/global_costmap/costmap"
COSTMAP_UPDATES_TOPIC = "/move_base/global_costmap/costmap_updates"

LETHAL = 253          # 254 = 障碍本身，253 = 内切膨胀（规划器眼里同样是走不了）


def say(msg):
    print(msg, flush=True)


def head(title):
    say("")
    say("=== %s ===" % title)


# ----------------------------------------------------------------- 航点文件
def load_waypoints(path, wanted=None):
    """读 waterplus 的航点 xml，返回 {name: (x, y)}。"""
    import xml.etree.ElementTree as ET

    out = {}
    root = ET.parse(path).getroot()
    for wp in root.iter("Waypoint"):
        name_el = wp.find("Name")
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        if wanted and name != wanted:
            continue
        try:
            x = float(wp.find("Pos_x").text)
            y = float(wp.find("Pos_y").text)
        except Exception:
            continue
        out[name] = (x, y)
    return out


# ----------------------------------------------------------------- 地图工具
class StaticMap(object):
    """静态地图：世界坐标<->栅格，以及「离最近墙有多远」。"""

    def __init__(self, grid):
        self.msg = grid
        self.w = grid.info.width
        self.h = grid.info.height
        self.res = grid.info.resolution
        self.ox = grid.info.origin.position.x
        self.oy = grid.info.origin.position.y
        data = np.array(grid.data, dtype=np.int16).reshape(self.h, self.w)
        self.data = data
        ys, xs = np.nonzero(data >= 65)          # 黑格 = 占用
        if len(xs) > 200000:                     # 地图很大时抽稀，0.1 m 精度足够
            xs, ys = xs[::2], ys[::2]
        pts = np.stack([
            self.ox + (xs + 0.5) * self.res,
            self.oy + (self.h - 0.5 - ys) * self.res,
        ], axis=1)
        from scipy.spatial import cKDTree

        self.tree = cKDTree(pts) if len(pts) else None

    def world_to_pix(self, x, y):
        c = int((x - self.ox) / self.res)
        r = self.h - 1 - int((y - self.oy) / self.res)
        if 0 <= c < self.w and 0 <= r < self.h:
            return c, r
        return None

    def value_at(self, x, y):
        pix = self.world_to_pix(x, y)
        if pix is None:
            return None
        return int(self.data[pix[1], pix[0]])

    def distance_to_wall(self, points):
        """points: Nx2 世界坐标 -> 每个点离最近占用格的距离（m）。"""
        if self.tree is None or len(points) == 0:
            return np.array([])
        d, _ = self.tree.query(points)
        return d


class Costmap(object):
    """全局代价地图（含膨胀）。先收一帧全量，再用 updates 补。"""

    def __init__(self, timeout=12.0):
        self.msg = None
        self.data = None
        self._sub_full = rospy.Subscriber(COSTMAP_TOPIC, OccupancyGrid, self._on_full)
        self._sub_upd = rospy.Subscriber(
            COSTMAP_UPDATES_TOPIC, OccupancyGrid, self._on_upd
        )
        t0 = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < timeout:
            if self.msg is not None:
                break
            rospy.sleep(0.2)

    def _on_full(self, msg):
        self.msg = msg
        self.data = np.array(msg.data, dtype=np.int16)

    def _on_upd(self, msg):
        if self.msg is None or len(msg.data) != len(self.data):
            return
        upd = np.array(msg.data, dtype=np.int16)
        mask = upd >= 0
        self.data[mask] = upd[mask]

    @property
    def ready(self):
        return self.data is not None

    def cost_at(self, x, y):
        info = self.msg.info
        c = int((x - info.origin.position.x) / info.resolution)
        r = info.height - 1 - int((y - info.origin.position.y) / info.resolution)
        if 0 <= c < info.width and 0 <= r < info.height:
            return int(self.data[r * info.width + c])
        return None

    def neighborhood(self, x, y, radius):
        """返回 (半径内格数, 其中 >=253 的格数, 半径内最高代价)。"""
        info = self.msg.info
        res = info.resolution
        c = int((x - info.origin.position.x) / res)
        r = info.height - 1 - int((y - info.origin.position.y) / res)
        rad = int(max(1, radius / res))
        r0, r1 = max(0, r - rad), min(info.height, r + rad + 1)
        c0, c1 = max(0, c - rad), min(info.width, c + rad + 1)
        block = self.data.reshape(info.height, info.width)[r0:r1, c0:c1]
        yy, xx = np.mgrid[r0:r1, c0:c1]
        ring = ((yy - r) ** 2 + (xx - c) ** 2) <= rad * rad
        vals = block[ring]
        if vals.size == 0:
            return 0, 0, 0
        return int(vals.size), int((vals >= LETHAL).sum()), int(vals.max())


# ----------------------------------------------------------------- 激光
def scan_points_in_map(scan, tf_buffer, target="map", max_range=6.0):
    """把一帧 LaserScan 的端点投到 map 系，返回 Nx4: x, y, r, angle。"""
    try:
        tr = tf_buffer.lookup_transform(
            target, scan.header.frame_id, scan.header.stamp, rospy.Duration(1.0)
        )
    except Exception as exc:  # noqa: BLE001
        rospy.logwarn("拿不到 %s -> %s 的 TF：%s", scan.header.frame_id, target, exc)
        return np.zeros((0, 4))
    q = tr.transform.rotation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
    tx = tr.transform.translation.x
    ty = tr.transform.translation.y
    ang = scan.angle_min
    pts = []
    for r in scan.ranges:
        a = ang
        ang += scan.angle_increment
        if not np.isfinite(r) or r < scan.range_min or r > min(scan.range_max, max_range):
            continue
        pts.append((tx + r * math.cos(a + yaw), ty + r * math.sin(a + yaw), r, a))
    return np.array(pts) if pts else np.zeros((0, 4))


def tf_pose(tf_buffer, target="map", source="base_footprint"):
    tr = tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(3.0))
    q = tr.transform.rotation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
    return tr.transform.translation.x, tr.transform.translation.y, yaw


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waypoints", default=DEFAULT_WAYPOINTS)
    ap.add_argument("--goal", default="", help="只检查这个航点")
    ap.add_argument("--scan-limit", type=float, default=6.0)
    args = ap.parse_args()

    rospy.init_node("nav_doctor", anonymous=True)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
    tf2_ros.TransformListener(tf_buffer)

    say("导航体检开始（只读，不动车）")
    problems = []

    # ---- 1. 地图 ----
    head("1) 静态地图")
    try:
        amap = rospy.wait_for_message("/map", OccupancyGrid, timeout=15.0)
    except rospy.ROSException:
        say("!! 收不到 /map：map_server 没起来？")
        return 1
    smap = StaticMap(amap)
    say("尺寸 %d x %d，%.3f m/格，原点 (%.2f, %.2f)  ->  %.1f m x %.1f m"
        % (smap.w, smap.h, smap.res, smap.ox, smap.oy,
           smap.w * smap.res, smap.h * smap.res))
    if smap.w * smap.res > 60.0:
        say("   ! 地图偏大：全局代价地图每轮都要做一次整图膨胀，")
        say("     日志里会看到 Map update loop missed its desired rate（实测 2.6~4.4 s）；")
        say("     规划器拿到的代价地图总是慢半拍，清代价地图也来不及生效。")

    # ---- 2. 定位 ----
    head("2) 定位 / 当前位姿")
    try:
        px, py, pyaw = tf_pose(tf_buffer)
    except Exception as exc:  # noqa: BLE001
        say("!! 拿不到 map->base_footprint 的 TF：%s" % exc)
        say("   amcl 没在跑？odom->base_footprint 断了？先在 RViz 里点 2D Pose Estimate")
        return 1
    say("TF 位姿   : x=%.2f y=%.2f yaw=%.2f (%.0f 度)"
        % (px, py, pyaw, math.degrees(pyaw)))
    try:
        ap_msg = rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=5.0)
        ax = ap_msg.pose.pose.position.x
        ay = ap_msg.pose.pose.position.y
        say("amcl_pose : x=%.2f y=%.2f（与 TF 差 %.2f m）"
            % (ax, ay, math.hypot(ax - px, ay - py)))
    except rospy.ROSException:
        say("amcl_pose : 收不到（AMCL 没起来？）")
        problems.append("AMCL 无 /amcl_pose")
    mval = smap.value_at(px, py)
    say("地图取值  : %s（>=65 = 墙/占用，255 = 未知，-1 = 越界）"
        % ("越界" if mval is None else mval))
    if mval is not None and mval >= 65:
        problems.append("机器人当前位姿落在地图的墙/未知区里 -> 定位错了")
        say("   ! 位姿落在墙上：基本可以确定是定位问题（初值不对 / 地图和现场不一致）")

    # ---- 3. 激光 / 定位吻合度 ----
    head("3) 激光端点 vs 静态地图的墙（判断定位偏没偏）")
    try:
        scan = rospy.wait_for_message("/scan", LaserScan, timeout=15.0)
    except rospy.ROSException:
        say("!! 收不到 /scan：雷达没插/没数据（先看 ls -l /dev/rplidar）")
        problems.append("没有 /scan")
        scan = None
    if scan is not None:
        pts = scan_points_in_map(scan, tf_buffer, max_range=args.scan_limit)
        say("本帧有效点 %d 个（量程 %.2f~%.2f m，截到 %.1f m）"
            % (len(pts), scan.range_min, scan.range_max, args.scan_limit))
        if len(pts):
            order = np.argsort(pts[:, 2])
            say("最近的 5 个点（角度是机器人坐标系下的方向）：")
            for i in order[:5]:
                say("   r=%.2f m  angle=%.1f 度"
                    % (pts[i, 2], math.degrees(pts[i, 3]) % 360.0))
            d = smap.distance_to_wall(pts[:, :2])
            if len(d):
                bad = float((d > 0.30).mean())
                say("激光端点离地图最近墙的距离：中位 %.2f m，平均 %.2f m，超过 0.30 m 的占 %.0f%%"
                    % (float(np.median(d)), float(d.mean()), bad * 100.0))
                if np.median(d) > 0.25 or bad > 0.5:
                    problems.append("激光与地图对不上（定位偏了）")
                    say("   ! 激光端点大多飘在空地中间 -> 定位和地图对不上。")
                    say("     这时障碍层会把墙标到空地上，规划器自然找不到路。")
                    say("     处理：先用 RViz 2D Pose Estimate 对准（并让任务别覆盖初值），")
                    say("     或检查/重做地图。")
                else:
                    say("   OK 激光贴着地图的墙，定位和地图基本吻合")

    # ---- 4. 全局代价地图 ----
    head("4) 全局代价地图（含膨胀）")
    cm = Costmap()
    if not cm.ready:
        say("!! 收不到 %s（move_base 没起来？）" % COSTMAP_TOPIC)
        problems.append("没有全局代价地图")
    else:
        info = cm.msg.info
        say("尺寸 %d x %d，%.3f m/格" % (info.width, info.height, info.resolution))
        c0 = cm.cost_at(px, py)
        say("机器人格代价 : %s（>=%d 就等于规划器眼里的障碍）"
            % ("越界" if c0 is None else c0, LETHAL))
        for rad in (0.25, 0.5, 0.75):
            tot, bad, mx = cm.neighborhood(px, py, rad)
            say("   周围 %.2f m：%d 格，其中 >=%d 的 %d 格（%.0f%%），最高 %d"
                % (rad, tot, LETHAL, bad, 100.0 * bad / max(1, tot), mx))
        tot, bad, mx = cm.neighborhood(px, py, 0.5)
        if tot and bad / float(tot) > 0.5:
            problems.append("机器人被实时障碍围住")
            say("   ! 0.5 m 内超过一半是致命/内切格 -> 机器人被围住了（人/家具/自身体），")
            say("     A* 一步都迈不出去，就会出现 Failed to get a plan.。")
        if c0 is not None and c0 >= LETHAL:
            problems.append("机器人所在格是致命格")

    # ---- 5. 航点与规划 ----
    head("5) 航点代价 + make_plan 试算（用当前位姿当起点）")
    try:
        wps = load_waypoints(args.waypoints, args.goal or None)
    except Exception as exc:  # noqa: BLE001
        say("!! 读航点文件失败：%s" % exc)
        wps = {}
    if not wps:
        say("!! 没有任何航点（文件 %s）" % args.waypoints)
    make_plan = None
    try:
        rospy.wait_for_service("/move_base/make_plan", timeout=10.0)
        make_plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
    except rospy.ROSException:
        say("!! /move_base/make_plan 服务不可用，跳过试算")

    start = PoseStamped()
    start.header.frame_id = "map"
    start.header.stamp = rospy.Time.now()
    start.pose.position.x = px
    start.pose.position.y = py
    start.pose.orientation.w = 1.0

    for name, (wx, wy) in sorted(wps.items()):
        line = "%-12s x=%6.2f y=%6.2f" % (name, wx, wy)
        dv = smap.value_at(wx, wy)
        line += "  地图=%-5s" % ("越界" if dv is None else dv)
        if cm.ready:
            cost = cm.cost_at(wx, wy)
            line += " 代价=%-5s" % ("越界" if cost is None else cost)
        dist = smap.distance_to_wall(np.array([[wx, wy]]))
        if len(dist):
            line += " 离墙=%.2f m" % float(dist[0])
        say(line)
        if dv is not None and dv >= 65:
            say("   ! 这个航点落在地图的墙/未知区里 -> 永远规划不到，需要重新标点")
        if cm.ready:
            cost = cm.cost_at(wx, wy)
            if cost is not None and cost >= LETHAL:
                say("   ! 目标格代价 >=%d：规划器不会往里走（膨胀吃掉了这个点）" % LETHAL)
        if make_plan is not None:
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = wx
            goal.pose.position.y = wy
            goal.pose.orientation.w = 1.0
            try:
                resp = make_plan(start, goal, 0.10)
                n = len(resp.plan.poses)
                if n:
                    say("   make_plan: OK %d 个路径点（约 %.1f m）" % (n, n * 0.05))
                else:
                    say("   make_plan: 空路径（就是从当前位姿规划不过去）")
                    problems.append("make_plan 到 %s 失败" % name)
            except Exception as exc:  # noqa: BLE001
                say("   make_plan: 调用失败 %s" % exc)

    # ---- 结论 ----
    head("结论")
    if not problems:
        say("没查出问题：地图/定位/激光/代价地图/规划都正常。")
        say("如果 ⑯ 还是失败，那就是跑的时候刚好有障碍（人走动）——")
        say("请对照失败时刻的日志，再在现场跑一次本脚本。")
        return 0
    for p in problems:
        say("- %s" % p)
    say("")
    say("怎么读：")
    say("  1) 只有「激光与地图对不上」        -> 定位问题（2D Pose Estimate / 别让任务覆盖初值 / 重做地图）")
    say("  2) 只有「被实时障碍围住」          -> 现场清障，或把膨胀半径调小")
    say("  3) 只有某航点「代价>=253 / 落墙里」-> 重新标那个航点（离墙 >= 0.75 m）")
    say("  4) 规划器本身很慢（Map update loop missed）-> 把大地图裁小")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(130)
