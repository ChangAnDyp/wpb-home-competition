#!/usr/bin/env python3
# coding: utf-8
"""Kinect v2 画面流看门狗：把"掉一次流"限制成几十秒，而不是毁掉整轮比赛。

现场看到的两种掉线（① 刚起栈跑任务就掉；② 注册完开始找人时掉）最后都收敛到
同一个故障形态：``/kinect2/qhd/image_color_rect`` 突然不再有新帧，节点还在、
话题还在、设备在 USB 上也枚举正常。老版本的重启链路救不回来，原因有三条，
都是实测（2026-10-09 的 rosout + dmesg + 节点自身日志）踩出来的：

1. **只重载 bridge 不够**。``rosnode kill /kinect2_bridge`` 只是让 roslaunch
   重新执行一次 ``nodelet load``；相机设备其实是被 nodelet **manager**
   （``/kinect2`` 这个进程）持有的。manager 卡住时（libfreenect2 传输线程没退
   干净、unload 阻塞），新的 load 请求永远执行不完——日志里的表现就是 kill
   掉 bridge 之后再没有任何 ``Kinect2Bridge::initialize``，画面永远不回来。
2. **USB 复位时机错了**。设备还被 bridge 持有着的时候做 port reset，会让旧
   handle 的 interface claim 失效，驱动随即报
   ``usbfs: process ... did not claim interface 0 before use``，之后所有传输
   被内核拒掉——等于把"还能救"变成"救不回来"。正确顺序是：先让持设备的进程
   真的退出（manager 死掉）→ 再复位 USB → 最后让它重新打开设备。
3. **恢复预算用完就永久放弃**。老版本 4 次重载 + 3 次 USB 复位之后只刷日志、
   再也不尝试，现场表现就是"卡死后一直没画面，只能人工重启整栈"。

所以这一版把恢复做成一条能自愈的梯子，并且**只有在确认持设备的进程已经退出
之后**才动 USB：

    ① 重载 bridge nodelet（最快，~20s，救"软件抖动"）
    ② 再重载一次（偶发的一次失败很常见）
    ③ 重启整个 nodelet manager（释放设备句柄，救"manager 卡死"）
    ④ USB 软拔插（sysfs ``authorized`` 0→1，需要 root；没权限时退化为
       USBDEVFS_RESET）—— 只在设备没人持有时执行
    ⑤ 每一轮都比上一轮等得更久，但**永不放弃**；同时把"需要人工处理"的话
       打在最显眼的位置

另外两个细节：

* 看门狗常驻订阅彩色图（掉流检测）+ 便宜的 ``/kinect2/sd/camera_info``
  （把桥的"有订阅者"状态钉住）。厂商 bridge 的 ``callbackStatus`` 会在订阅者
  数量变化时 ``device->stop()`` + ``startStreams()``，而每次 stop/start 都是
  一次掉流风险。
* 如果本栈的 bridge 已经不在 master 上（上一轮栈的看门狗残留、或栈已经被关
  掉），本节点**不再动 USB**，避免误伤正在跑的另一套栈。
"""

import fcntl
import glob
import os
import subprocess
import threading
import time

import rospy
from sensor_msgs.msg import CameraInfo, Image

# linux/usbdevice_fs.h: USBDEVFS_RESET = _IO('U', 20)
USBDEVFS_RESET = (ord("U") << 8) | 20


def find_usb_device(vendor_id, product_id, sysfs_root="/sys/bus/usb/devices"):
    """按 idVendor/idProduct 在 sysfs 里定位设备，返回 (busnum, devnum, sysfs_name)。

    不能按 USB 端口号找：现场换过口、也换过控制器，只有 VID:PID 是稳定的。
    """
    want = (str(vendor_id).lower(), str(product_id).lower())
    for path in sorted(glob.glob(os.path.join(sysfs_root, "*"))):
        try:
            with open(os.path.join(path, "idVendor")) as handle:
                vendor = handle.read().strip().lower()
            with open(os.path.join(path, "idProduct")) as handle:
                product = handle.read().strip().lower()
        except OSError:
            continue
        if (vendor, product) != want:
            continue
        try:
            with open(os.path.join(path, "busnum")) as handle:
                busnum = int(handle.read().strip())
            with open(os.path.join(path, "devnum")) as handle:
                devnum = int(handle.read().strip())
        except (OSError, ValueError):
            continue
        return busnum, devnum, os.path.basename(path)
    return None


def devnode_for_sysfs(
    sysfs_name, sysfs_root="/sys/bus/usb/devices", dev_root="/dev/bus/usb"
):
    """把 sysfs 里的设备名（例如 ``4-2``）映射成 /dev/bus/usb/BBB/DDD。

    不用 ``busnum``/``devnum`` 直接拼路径：现场日志里出现过拼出
    ``/dev/bus/usb/004/002`` 而设备实际在 ``004/004`` 的情况（内核会复用 devnum，
    复位瞬间读到的值可能已过期）。这里改成拿 sysfs 的 ``dev`` 属性（major:minor）
    去 ``/dev/bus/usb/*/*`` 逐个 ``os.stat`` 比对 ``st_rdev``，只认真正对得上的
    节点。拿不到就返回 None，调用方宁可不复位，也不要去复位别的设备。
    """
    dev_file = os.path.join(sysfs_root, sysfs_name, "dev")
    try:
        with open(dev_file) as handle:
            major_text, minor_text = handle.read().strip().split(":")
        want = (int(major_text), int(minor_text))
    except (OSError, ValueError):
        return None

    for path in sorted(glob.glob(os.path.join(dev_root, "*", "*"))):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if (os.major(stat.st_rdev), os.minor(stat.st_rdev)) == want:
            return path
    return None


def device_holders(devnode):
    """列出当前打开着某个 usb devnode 的进程 ``(pid, comm)``。

    动 USB 之前必须先确认没人持有设备：bridge 还开着设备时做 port reset 会把
    它的 interface claim 打坏（见模块文档第 2 条）。
    """
    if not devnode:
        return []
    holders = []
    for proc_dir in glob.glob("/proc/[0-9]*"):
        pid = os.path.basename(proc_dir)
        fd_dir = os.path.join(proc_dir, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target == devnode:
                try:
                    with open(os.path.join(proc_dir, "comm")) as handle:
                        comm = handle.read().strip()
                except OSError:
                    comm = "?"
                holders.append((pid, comm))
                break
    return holders


def power_cycle_usb_port(sysfs_name, sysfs_root="/sys/bus/usb/devices"):
    """通过 sysfs ``authorized`` 做一次“断电重枚举”，比 USBDEVFS_RESET 更彻底。

    ``authorized`` 0→1 会让设备从总线上摘掉再加回来（等价一次软件拔插），这是
    纯软件能做的最后一级恢复。``authorized`` 是 ``0644 root:root``，所以默认
    需要 root；没权限时返回 False，调用方回退到 USBDEVFS_RESET。
    """
    authorized = os.path.join(sysfs_root, sysfs_name, "authorized")
    if not os.path.exists(authorized):
        return False, "%s not present" % authorized
    try:
        with open(authorized, "w") as handle:
            handle.write("0")
    except OSError as exc:
        return False, "cannot deauthorize %s (%s): needs root" % (authorized, exc)
    # 摘掉设备后必须保证会加回来，否则相机会永久消失。
    time.sleep(1.0)
    try:
        with open(authorized, "w") as handle:
            handle.write("1")
    except OSError as exc:
        return False, (
            "deauthorized %s but could not re-authorize it (%s); "
            "the Kinect must be replugged by hand" % (authorized, exc)
        )
    return True, "power-cycled %s via sysfs authorized" % sysfs_name


def reset_usb_device(
    vendor_id,
    product_id,
    sysfs_root="/sys/bus/usb/devices",
    dev_root="/dev/bus/usb",
):
    """尽最大努力在软件层“拔插”一次 Kinect。

    依次尝试由强到弱的两级：
      1. sysfs ``authorized`` 0→1（真·断电重枚举，需要 root）；
      2. ``USBDEVFS_RESET`` ioctl（udev 规则给了 0666，不需要 root）。
    返回 (ok, message)。
    """
    found = find_usb_device(vendor_id, product_id, sysfs_root=sysfs_root)
    if found is None:
        return False, "device %s:%s not present in sysfs" % (vendor_id, product_id)
    _busnum, _devnum, sysfs_name = found

    devnode = devnode_for_sysfs(sysfs_name, sysfs_root=sysfs_root, dev_root=dev_root)
    if devnode:
        holders = device_holders(devnode)
        if holders:
            return False, (
                "refusing to reset %s: still held by %s"
                % (devnode, ", ".join("%s(%s)" % (pid, comm) for pid, comm in holders[:4]))
            )

    ok, message = power_cycle_usb_port(sysfs_name, sysfs_root=sysfs_root)
    if ok:
        return True, message

    # 记录降级原因，现场日志要能看出“为什么没用更强的那一级”。
    fallback_reason = message
    if devnode is None:
        return False, (
            "cannot resolve the /dev node for %s (%s); "
            "refusing to guess (wrong node = broken claim on another device)"
            % (sysfs_name, fallback_reason)
        )
    try:
        fd = os.open(devnode, os.O_WRONLY)
    except OSError as exc:
        return False, "cannot open %s (%s); check udev MODE for %s:%s" % (
            devnode,
            exc,
            vendor_id,
            product_id,
        )
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    except OSError as exc:
        return False, "USBDEVFS_RESET on %s (%s) failed: %s" % (
            devnode,
            sysfs_name,
            exc,
        )
    finally:
        os.close(fd)
    return True, "USBDEVFS_RESET on %s (%s); sysfs authorized unavailable: %s" % (
        devnode,
        sysfs_name,
        fallback_reason,
    )


class Kinect2StreamWatchdog(object):
    """盯住彩色流 + 掉流时按梯子自愈。"""

    def __init__(self):
        self.image_topic = str(
            rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        )
        # 多久没有新帧就认为停发。太短会在高负载时误杀，6 秒足够区分抖动和停发。
        self.stall_seconds = max(2.0, float(rospy.get_param("~stall_seconds", 6.0)))
        # 启动宽限期：bridge 起来、YOLO/ReID 抢 GPU 的这段时间本来就会丢帧。
        self.warmup_seconds = max(0.0, float(rospy.get_param("~warmup_seconds", 20.0)))
        self.check_period = max(0.2, float(rospy.get_param("~check_period", 1.0)))
        self.recovery_enabled = bool(rospy.get_param("~restart_enabled", True))

        # ---- 节点名（恢复的对象）------------------------------------------
        self.bridge_node_name = str(rospy.get_param("~bridge_node_name", "/kinect2_bridge"))
        # nodelet manager：真正持有相机设备的进程。
        self.manager_node_name = str(rospy.get_param("~manager_node_name", "/kinect2"))
        # 一起加载在同一个 manager 里的其它 nodelet（点云），重启 manager 时顺手确认。
        self.extra_nodelet_names = [
            str(name)
            for name in rospy.get_param(
                "~extra_nodelet_names", ["/kinect2_points_xyzrgb_qhd"]
            )
        ]

        # ---- 梯子节奏 ------------------------------------------------------
        # 每次动作之后给画面留多少时间回来看（不到就升级下一级）。
        self.verify_seconds = max(5.0, float(rospy.get_param("~verify_seconds", 20.0)))
        # 同一级连续两次动作之间的最短间隔（重载 bridge 用）。
        self.bridge_restart_cooldown = max(
            1.0, float(rospy.get_param("~restart_cooldown", 25.0))
        )
        # 每轮梯子重复几次“重载 bridge”再去动 manager。
        self.bridge_restarts_per_round = max(
            1, int(rospy.get_param("~bridge_restarts_per_round", 2))
        )
        # manager 收到 kill 之后最多等多久；还不消失就 SIGKILL。
        self.manager_kill_timeout = max(
            2.0, float(rospy.get_param("~manager_kill_timeout", 8.0))
        )
        # 重启 manager 之后给 roslaunch respawn + 相机重新初始化的时间。
        self.manager_verify_seconds = max(
            self.verify_seconds, float(rospy.get_param("~manager_verify_seconds", 30.0))
        )
        # 一轮梯子（重载×N → 重启 manager → USB 软拔插）失败后等多久再从头来。
        # 不设上限、不放弃：现场只要还有电，就还有机会自己爬起来。
        self.round_retry_seconds = max(
            20.0, float(rospy.get_param("~round_retry_seconds", 90.0))
        )
        # 画面稳定这么久之后，把梯子等级清零，让一次长任务能扛过多次掉线。
        self.recovery_reset_seconds = max(
            10.0, float(rospy.get_param("~recovery_reset_seconds", 30.0))
        )
        # 动 USB 之前先等设备被释放多久（防止刚 respawn 的 bridge 又抓着设备）。
        self.devnode_free_seconds = max(
            0.5, float(rospy.get_param("~devnode_free_seconds", 3.0))
        )

        # ---- 二级恢复：USB 软拔插 ------------------------------------------
        self.usb_reset_enabled = bool(rospy.get_param("~usb_reset_enabled", True))
        self.usb_vendor_id = str(rospy.get_param("~usb_vendor_id", "045e"))
        self.usb_product_id = str(rospy.get_param("~usb_product_id", "02c4"))
        # 有 root 权限的辅助脚本（sudoers 里给它 NOPASSWD）。存在就用它做
        # sysfs authorized 软拔插；不存在就退化成 USBDEVFS_RESET。
        self.usb_power_cycle_command = str(
            rospy.get_param(
                "~usb_power_cycle_command", "/usr/local/sbin/kinect2-usb-powercycle"
            )
        )
        # 设备整机掉线（USB disconnect，不是卡死）时，等内核重新枚举它最多等多久。
        # 现场实测：xHCI 上掉一次到重新枚举，中间空了 54 秒（15:58、16:00、16:09
        # 三次都是这个量级），所以这个值要给够。
        self.usb_replug_wait_seconds = max(
            5.0, float(rospy.get_param("~usb_replug_wait_seconds", 60.0))
        )
        self.usb_reset_settle_seconds = max(
            0.5, float(rospy.get_param("~usb_reset_settle_seconds", 3.0))
        )

        # ---- 状态 ----------------------------------------------------------
        self.lock = threading.Lock()
        self.last_message_time = None
        self.message_count = 0
        self.keepalive_count = 0
        self.round_index = 0          # 第几轮梯子（0 = 第一轮）
        self.attempts_this_stall = 0  # 本轮已经做过几个动作（1..N）
        self.actions_total = 0
        self.manager_restart_count = 0
        self.usb_cycle_count = 0
        self.bridge_restart_count = 0
        self.started_time = time.time()
        self.grace_until = self.started_time + self.warmup_seconds
        self.next_action_time = 0.0
        self.last_action_time = 0.0
        self.escalated = False

        self.sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24
        )
        # 便宜的深度保活订阅：让 bridge 的“有订阅者”集合在整轮任务里保持不变，
        # 避免 callbackStatus 触发 device->stop()/startStreams()（每次都是一次风险）。
        self.keepalive_sub = None
        self.keepalive_topic = str(
            rospy.get_param("~keepalive_depth_topic", "/kinect2/sd/camera_info")
        )
        if self.keepalive_topic:
            self.keepalive_sub = rospy.Subscriber(
                self.keepalive_topic, CameraInfo, self.keepalive_callback, queue_size=1
            )

        rospy.loginfo(
            "Kinect2 stream watchdog armed: topic=%s stall=%.1fs warmup=%.1fs "
            "bridge=%s manager=%s keepalive=%s usb_reset=%s(%s:%s) round_retry=%.0fs",
            self.image_topic,
            self.stall_seconds,
            self.warmup_seconds,
            self.bridge_node_name,
            self.manager_node_name,
            self.keepalive_topic or "off",
            self.usb_reset_enabled,
            self.usb_vendor_id,
            self.usb_product_id,
            self.round_retry_seconds,
        )

    # ------------------------------------------------------------------
    # 订阅回调 / 状态
    # ------------------------------------------------------------------
    def image_callback(self, _msg):
        with self.lock:
            self.last_message_time = time.time()
            self.message_count += 1

    def keepalive_callback(self, _msg):
        with self.lock:
            self.keepalive_count += 1

    def snapshot(self):
        with self.lock:
            return self.last_message_time, self.message_count

    def advertised(self):
        try:
            published = rospy.get_published_topics("/")
        except Exception as exc:  # noqa: BLE001 - 话题查询失败不该让看门狗退出
            rospy.logwarn_throttle(30.0, "Kinect2 watchdog cannot list topics: %s", exc)
            return True
        normalized = self.image_topic.rstrip("/") or "/"
        for name, _topic_type in published:
            if (name.rstrip("/") or "/") == normalized:
                return True
        return False

    def usb_device_present(self):
        """Kinect 现在还在 USB 总线上吗（按 VID:PID 找）。"""
        try:
            busnum, devnum, _name = find_usb_device(
                self.usb_vendor_id, self.usb_product_id
            )
        except Exception:  # noqa: BLE001 - 查不到就当不在，交给下层处理
            return False
        return busnum is not None and devnum is not None

    def usb_devnode(self):
        found = find_usb_device(self.usb_vendor_id, self.usb_product_id)
        if found is None:
            return None, None
        return devnode_for_sysfs(found[2]), found[2]

    def running_nodes(self):
        """``rosnode list`` 的结果（拿不到返回 None）。"""
        try:
            result = subprocess.run(
                ["rosnode", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=8.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            rospy.logwarn_throttle(30.0, "看门狗跑不了 rosnode list: %s", exc)
            return None
        if result.returncode != 0:
            return None
        return result.stdout.decode("utf-8", errors="replace").split()

    def node_registered(self, name):
        """节点还在 master 上吗（查不到时保守认为在）。"""
        nodes = self.running_nodes()
        return True if nodes is None else name in nodes

    def own_stack_alive(self, nodes):
        """本栈的 bridge/manager 还在 master 上吗。

        不在就说明这不是"我们的栈"掉流（例如上一轮栈的看门狗残留、或栈已经被
        关掉），此时绝不能去动 USB：否则会把另一套正在跑的栈的相机 claim 打坏。
        """
        if nodes is None:
            # 连 ``rosnode list`` 都跑不通，说明自己连的 master 已经没了
            # （典型：上一轮的看门狗残留）。这时绝不能去动 USB。
            return False
        for name in (self.bridge_node_name, self.manager_node_name):
            if name and name in nodes:
                return True
        return False

    def kill_node(self, name, timeout=None):
        """``rosnode kill``，必要时 SIGKILL（manager 卡死时 SIGINT 会被忽略）。"""
        limit = self.manager_kill_timeout if timeout is None else max(1.0, float(timeout))
        try:
            result = subprocess.run(
                ["rosnode", "kill", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(5.0, limit + 5.0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            rospy.logerr("跑 rosnode kill %s 失败：%s", name, exc)
            return False
        output = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode != 0:
            rospy.logwarn("rosnode kill %s 返回 %d：%s", name, result.returncode, output)
        else:
            rospy.logwarn("rosnode kill %s -> %s", name, output or "ok")

        deadline = time.time() + limit
        while time.time() < deadline and not rospy.is_shutdown():
            nodes = self.running_nodes()
            if nodes is not None and name not in nodes:
                return True
            rospy.sleep(0.5)
        self.force_kill_node(name)
        return False

    def force_kill_node(self, name):
        """manager 卡死时 ``rosnode kill`` 可能只发 SIGINT 就返回，需要补一刀。"""
        try:
            result = subprocess.run(
                ["rosnode", "info", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=8.0,
            )
        except (OSError, subprocess.SubprocessError):
            return
        pid = None
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("Pid:"):
                try:
                    pid = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pid = None
                break
        if not pid:
            return
        rospy.logerr(
            "%s 没能在 %.1fs 内退出（nodelet manager 卡死的典型症状），补 SIGKILL 到 pid %d",
            name,
            self.manager_kill_timeout,
            pid,
        )
        try:
            os.kill(pid, 9)
        except OSError as exc:
            rospy.logerr("SIGKILL %s(pid %d) 失败：%s", name, pid, exc)

    # ------------------------------------------------------------------
    # 恢复动作
    # ------------------------------------------------------------------
    def restart_bridge(self):
        """① / ② 重载 bridge nodelet（最快的一级）。"""
        self.bridge_restart_count += 1
        rospy.logerr(
            "Kinect2 画面停发：重载 %s（第 %d 次），靠 respawn 重新 nodelet load 打开设备",
            self.bridge_node_name,
            self.bridge_restart_count,
        )
        self.kill_node(self.bridge_node_name, timeout=5.0)

    def restart_manager(self):
        """③ 重启整个 nodelet manager —— 真正释放相机设备句柄的那一级。"""
        self.manager_restart_count += 1
        rospy.logerr(
            "Kinect2 画面停发：重载 bridge 无效，重启整个 nodelet manager %s"
            "（第 %d 次）。这一步会释放被卡住的相机句柄，之后靠 respawn 重新拉起"
            " manager 和 bridge —— 现场实测只有这一级能救回"
            "“话题在、0Hz、设备枚举正常”的卡死",
            self.manager_node_name,
            self.manager_restart_count,
        )
        # 顺序：先让持设备的进程退出，再谈别的。
        self.kill_node(self.manager_node_name)
        for name in self.extra_nodelet_names:
            if name and name != self.manager_node_name:
                self.kill_node(name, timeout=3.0)
        # manager 可能"从 master 注销了但进程没退出"（卡在内核/libusb 里）。
        # roslaunch 的 respawn 只看进程退出，所以这时必须补 SIGKILL —— 2026-10-09
        # 现场就是卡在这一步：设备一直被 pid 38320(nodelet) 占着，US 那级一路跳过。
        rospy.sleep(2.0)
        devnode, _sysfs_name = self.usb_devnode()
        if (
            devnode
            and device_holders(devnode)
            and not self.node_registered(self.manager_node_name)
        ):
            self.kill_device_holders(devnode, "重启 nodelet manager 后清理残留")
        rospy.logwarn(
            "manager 已重启；roslaunch 会按 respawn 重新拉起 manager + bridge"
            "（预计 %.0f 秒内画面恢复）",
            self.manager_verify_seconds,
        )

    def wait_device_free(self, devnode, timeout=None):
        """等设备被释放，返回 True=没人持有了。"""
        limit = (
            max(2.0, self.devnode_free_seconds * 4)
            if timeout is None
            else max(1.0, float(timeout))
        )
        deadline = time.time() + limit
        while not rospy.is_shutdown() and time.time() < deadline:
            holders = device_holders(devnode)
            if not holders:
                return True
            rospy.logwarn(
                "还在等相机设备被释放：%s 被 %s 持有",
                devnode,
                ", ".join("%s(%s)" % (pid, comm) for pid, comm in holders[:4]),
            )
            rospy.sleep(0.5)
        return not device_holders(devnode)

    def kill_device_holders(self, devnode, why):
        """把还抓着相机设备的进程 SIGKILL 掉（一般是卡死的 nodelet manager）。

        为什么必须这么干（2026-10-09 现场实测）：roslaunch 的 respawn 只在**子进程
        真的退出**时才触发。manager 收到 SIGINT 后会先从 master 注销（所以
        ``rosnode kill /kinect2`` 显示 killed、"Unknown node(s)"），但进程本身卡在
        libusb/freenect2 里不退出 —— 结果 roslaunch 永远不重启它、设备永远被占着，
        看门狗每次走到 USB 那级都因为"设备还被占用"而跳过，画面再也回不来。
        """
        holders = device_holders(devnode)
        if not holders:
            return True
        rospy.logerr(
            "%s：设备 %s 还被进程持有 %s，补 SIGKILL（不杀它 roslaunch 不会 respawn）",
            why,
            devnode,
            ", ".join("%s(%s)" % (pid, comm) for pid, comm in holders[:4]),
        )
        for pid, _comm in holders:
            try:
                os.kill(int(pid), 9)
            except OSError as exc:
                rospy.logerr("SIGKILL pid %s 失败：%s", pid, exc)
        return self.wait_device_free(devnode, timeout=8.0)

    def run_usb_power_cycle_helper(self):
        """优先用带 root 权限的辅助脚本做 sysfs ``authorized`` 软拔插。"""
        command = self.usb_power_cycle_command
        if not command:
            return False, "no power-cycle helper configured"
        if not os.path.exists(command):
            return False, "%s not installed" % command
        argv = [command]
        if os.geteuid() != 0:
            argv = ["sudo", "-n"] + argv
        try:
            result = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, "helper failed to run: %s" % exc
        output = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode != 0:
            return False, "helper rc=%d: %s" % (result.returncode, output or "no output")
        return True, output or "helper ok"

    def usb_power_cycle(self, round_index):
        """④ USB 软拔插：只在确定没人持有设备时执行。"""
        self.usb_cycle_count += 1
        devnode, sysfs_name = self.usb_devnode()
        rospy.logerr(
            "Kinect2 画面停发：升级到 USB 软拔插（第 %d 轮 / 第 %d 次），device=%s node=%s",
            round_index,
            self.usb_cycle_count,
            sysfs_name or "?",
            devnode or "?",
        )

        if devnode is None:
            # 设备整机掉线（dmesg 里是 `USB disconnect`）时会走到这里：不做 IOCTL，
            # 等内核重新枚举回来即可。
            rospy.logwarn(
                "目前读不到 Kinect 的 /dev 节点：可能整机掉线，等内核重新枚举"
                "（最多 %.0f 秒）",
                self.usb_replug_wait_seconds,
            )
            self.wait_for_usb_device()
            return

        if device_holders(devnode):
            # 还有进程抓着设备（多半是刚 respawn 的 bridge）：先把它停下来，
            # 否则 port reset 会把它的 interface claim 打坏，越救越死。
            rospy.logwarn("设备还被进程持有，先停掉 bridge/manager 再复位（顺序不能反）")
            self.kill_node(self.bridge_node_name, timeout=4.0)
            self.kill_node(self.manager_node_name)
            if not self.kill_device_holders(devnode, "USB 软拔插前置清理"):
                # 到这一步说明进程卡在内核里（D 状态）杀不掉。authorized 0→1 是
                # **内核级**操作，不要求进程先松开设备，也是解开这种卡死的唯一杠杆，
                # 所以照样往下做，只是把风险提示打出来。
                rospy.logerr(
                    "设备仍被占用且杀不掉（多半卡在内核态）：仍然继续软拔插 —— "
                    "authorized 0→1 会让内核拆掉这个设备，卡住的传输会以 ENODEV 返回，"
                    "之后残留进程才有机会退出"
                )

        ok, message = self.run_usb_power_cycle_helper()
        if not ok:
            rospy.logwarn("带 root 的软拔插不可用（%s），退化成 USBDEVFS_RESET", message)
            ok, message = reset_usb_device(self.usb_vendor_id, self.usb_product_id)
        if ok:
            rospy.logwarn("USB 软拔插成功：%s", message)
        else:
            rospy.logerr(
                "USB 软拔插失败：%s；如果连续几轮都失败，请人工断 Kinect 12V 电源重插",
                message,
            )
        # 复位后设备会重新枚举，必须让 bridge 重新打开它（新实例）。
        rospy.sleep(self.usb_reset_settle_seconds)
        self.wait_for_usb_device()
        self.kill_node(self.bridge_node_name, timeout=4.0)
        self.kill_node(self.manager_node_name)

    def wait_for_usb_device(self, timeout=None):
        """等设备重新被内核枚举出来。返回 True=回来了。"""
        limit = (
            self.usb_replug_wait_seconds if timeout is None else max(1.0, float(timeout))
        )
        deadline = time.time() + limit
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.usb_device_present():
                rospy.loginfo("Kinect 已重新枚举到总线上，继续恢复")
                rospy.sleep(self.usb_reset_settle_seconds)
                return True
            rospy.sleep(0.5)
        rospy.logerr(
            "等了 %.0f 秒 Kinect 还没回到 USB 总线上；如果是硬件/供电问题，"
            "这里等多久都不会好（请看 dmesg 里的 usb 4-2 记录）",
            limit,
        )
        return False

    # ------------------------------------------------------------------
    # 梯子调度
    # ------------------------------------------------------------------
    def note_healthy(self, now):
        if self.last_action_time <= 0.0:
            return
        if now - self.last_action_time < self.recovery_reset_seconds:
            return
        rospy.loginfo(
            "画面已稳定 %.0f 秒，看门狗梯子归零（累计：重载 %d 次 / 重启 manager %d 次 / "
            "USB 软拔插 %d 次）",
            now - self.last_action_time,
            self.bridge_restart_count,
            self.manager_restart_count,
            self.usb_cycle_count,
        )
        self.round_index = 0
        self.attempts_this_stall = 0
        self.last_action_time = 0.0
        self.escalated = False

    def perform_next_action(self, now, age, count):
        """按梯子做下一个动作；返回下一次允许动作的时间。"""
        if self.attempts_this_stall < self.bridge_restarts_per_round:
            self.attempts_this_stall += 1
            self.actions_total += 1
            self.last_action_time = now
            rospy.logerr(
                "Kinect2 流停发 %.1fs（收到 %d 帧）：第 %d 轮第 %d 步 = 重载 bridge nodelet",
                age,
                count,
                self.round_index + 1,
                self.attempts_this_stall,
            )
            self.restart_bridge()
            return now + min(self.bridge_restart_cooldown, self.verify_seconds)

        if self.attempts_this_stall == self.bridge_restarts_per_round:
            self.attempts_this_stall += 1
            self.actions_total += 1
            self.last_action_time = now
            self.restart_manager()
            return now + self.manager_verify_seconds

        # 已经过了 manager 那一级：USB 软拔插（每轮只做一次）。
        self.attempts_this_stall = 0
        self.round_index += 1
        self.last_action_time = now
        if self.usb_reset_enabled:
            self.usb_power_cycle(self.round_index)
        else:
            rospy.logerr("USB 软拔插被禁用（usb_reset_enabled=false），直接等下一轮")
        self.escalated = True
        rospy.logerr(
            "第 %d 轮恢复动作做完仍未恢复画面：%.0f 秒后重头再来。"
            "如果连续两轮都救不回来，请人工断 Kinect 12V 电源重插，"
            "或整栈重启（robot.sh stop 后重新起栈）",
            self.round_index,
            self.round_retry_seconds,
        )
        return now + self.round_retry_seconds

    def spin(self):
        rate = rospy.Rate(1.0 / self.check_period)
        while not rospy.is_shutdown():
            rate.sleep()
            now = time.time()
            if now < self.grace_until:
                continue

            last_message_time, count = self.snapshot()
            if last_message_time is None:
                # 还没收到过任何帧：可能是 bridge 还没起来，也可能是 USB 没插。
                if not self.advertised():
                    rospy.logwarn_throttle(
                        30.0,
                        "Kinect2 watchdog: %s is not advertised; bridge did not start",
                        self.image_topic,
                    )
                continue

            age = now - last_message_time
            if age <= self.stall_seconds:
                self.note_healthy(now)
                continue

            if not self.recovery_enabled:
                rospy.logerr_throttle(
                    30.0,
                    "Kinect2 stream stalled on %s: last frame %.1fs ago; restart disabled, "
                    "restart kinect2_bridge and run `rostopic hz %s`",
                    self.image_topic,
                    age,
                    self.image_topic,
                )
                continue

            if now < self.next_action_time:
                continue

            nodes = self.running_nodes()
            if not self.own_stack_alive(nodes):
                rospy.logerr_throttle(
                    60.0,
                    "本栈的 %s / %s 已不在 master 上（画面停发 %.1fs，收到的帧停在 %d）："
                    "这多半是别的会话/上一轮残留的节点，本节点不动 USB，"
                    "避免把正在跑的相机 claim 打坏",
                    self.bridge_node_name,
                    self.manager_node_name,
                    age,
                    count,
                )
                self.next_action_time = now + 60.0
                continue

            if not self.usb_device_present():
                # 整机掉线（dmesg 里 `USB disconnect`）：等内核枚举回来，不消耗梯子。
                rospy.logerr_throttle(
                    15.0,
                    "Kinect %s:%s 不在 USB 总线上（整机掉线，画面已停 %.1fs）；"
                    "等内核重新枚举，暂不重启 bridge",
                    self.usb_vendor_id,
                    self.usb_product_id,
                    age,
                )
                self.wait_for_usb_device()
                self.next_action_time = time.time() + 5.0
                continue

            self.next_action_time = self.perform_next_action(now, age, count)


def main():
    rospy.init_node("kinect2_stream_watchdog")
    watchdog = Kinect2StreamWatchdog()
    watchdog.spin()


if __name__ == "__main__":
    main()
