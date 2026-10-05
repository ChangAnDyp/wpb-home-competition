#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启智机器人比赛项目 · 全功能体检

分三层检查，每层都能单独跑：

  python3 feature_check.py            只做静态检查（快、安全，不动机器人）
  python3 feature_check.py --runtime  额外检查正在运行的栈（需先启动 robot.sh stack）

输出一张 PASS / FAIL / 需人工 的清单，方便逐个确认。
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "catkin_ws", "src")
PKG = os.path.join(SRC, "wpb_task1_owner_search")
VOICE = os.path.join(SRC, "offline_voice_bridge")

results = []          # (分组, 名称, 状态, 说明)


def add(group, name, ok, detail=""):
    if ok is True:
        status = "PASS"
    elif ok is False:
        status = "FAIL"
    else:
        status = "需人工"
    results.append((group, name, status, detail))


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "超时"
    except Exception as exc:
        return 1, str(exc)


def run_py(code, timeout=180, env_extra=None):
    """把代码写进临时文件再执行，避免 -c 的引号转义问题。"""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="fcheck_", dir="/tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(code)
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=env.get("FCHECK_CWD", "/tmp"))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "超时"
    except Exception as exc:
        return 1, str(exc)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def topic_hz(topic, seconds=5):
    code, out = run("timeout %d rostopic hz %s 2>&1" % (seconds + 2, topic), timeout=seconds + 6)
    m = re.search(r"average rate:\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None


def topic_hz_patient(topic, tries=5, seconds=5):
    """Kinect 在没有订阅者时会自动停设备，第一次连接要几秒才重新出图，
    所以这里多试几次，避免把“还没唤醒”误报成“无数据”。"""
    for _ in range(tries):
        hz = topic_hz(topic, seconds)
        if hz:
            return hz
        time.sleep(2)
    return None


def has_publisher(topic):
    code, out = run("timeout 10 rostopic info %s 2>&1" % topic, timeout=14)
    return "Publishers:" in out and "*" in out.split("Publishers:")[-1]


# ------------------------------------------------------------------ 1 环境
def check_environment():
    g = "环境与依赖"
    code, out = run("bash -lc 'source /opt/ros/noetic/setup.bash; echo $ROS_DISTRO'")
    add(g, "ROS Noetic 环境", out.strip() == "noetic", out.strip())

    code, out = run("bash -lc 'source /opt/ros/noetic/setup.bash; source %s/devel/setup.bash; "
                    "for p in wpb_task1_owner_search offline_voice_bridge perception_msgs "
                    "yoloworld_perception; do rospack find $p >/dev/null 2>&1 && echo -n ok || echo -n MISS:$p; done'" % os.path.join(HOME, "catkin_ws"))
    add(g, "4 个 ROS 包可找到", out.strip() == "okokokok", out.strip())

    mods = {
        "funasr": "语音识别(SenseVoice)",
        "insightface": "人脸识别",
        "ultralytics": "YOLO 检测/姿态",
        "torch": "深度学习框架",
        "cv2": "图像处理",
        "clip": "YOLO-World 文本编码器",
    }
    missing = []
    for mod, label in mods.items():
        code, out = run("python3 -c 'import %s' 2>&1" % mod)
        if code != 0:
            missing.append(mod)
    add(g, "Python 关键依赖 (6)", not missing,
        "全部就绪" if not missing else "缺: " + ", ".join(missing))

    # torchreid 需要把 third_party 加进 sys.path
    code, out = run("cd %s && python3 -c \"import sys; sys.path.insert(0,'third_party/deep-person-reid'); "
                    "import torchreid; print(torchreid.__version__)\" 2>&1" % PKG)
    add(g, "torchreid (人体 ReID)", "1.4.0" in out, out.strip().splitlines()[-1] if out.strip() else "")

    code, out = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1")
    add(g, "GPU 可用", "NVIDIA" in out, out.strip()[:60])


# ------------------------------------------------------------------ 2 模型
def check_models():
    g = "模型与权重"

    sv = os.path.join(HOME, "models", "SenseVoiceSmall")
    need = ["model.pt", "config.yaml", "tokens.json"]
    have = [f for f in need if os.path.exists(os.path.join(sv, f))]
    add(g, "SenseVoice 语音识别模型", len(have) == len(need),
        "%d/%d 个文件" % (len(have), len(need)))

    piper = os.path.join(HOME, "piper", "piper", "piper")
    voice = os.path.join(HOME, "piper", "voices", "zh_CN-huayan-medium.onnx")
    add(g, "Piper 语音合成", os.path.exists(piper) and os.path.exists(voice),
        "二进制+音色均在" if os.path.exists(piper) and os.path.exists(voice) else "缺文件")

    pose = os.path.join(HOME, "models", "yolo11n-pose.pt")
    add(g, "YOLO11n-pose 姿态模型", os.path.exists(pose),
        "%.1f MB" % (os.path.getsize(pose) / 1e6) if os.path.exists(pose) else "缺失")

    world = os.path.join(SRC, "yoloworld_perception", "models", "yolov8s-world-person.pt")
    add(g, "YOLO-World 行人检测权重", os.path.exists(world),
        "%.1f MB" % (os.path.getsize(world) / 1e6) if os.path.exists(world) else "缺失")

    clip_w = os.path.join(HOME, "weights", "clip", "ViT-B-32.pt")
    add(g, "CLIP 文本编码器权重", os.path.exists(clip_w),
        "已缓存（首次加载快）" if os.path.exists(clip_w) else "未缓存，首次启动需联网下载约 337MB")

    face = os.path.join(HOME, ".insightface", "models", "buffalo_sc")
    add(g, "InsightFace 人脸模型", os.path.isdir(face) and bool(os.listdir(face)),
        "buffalo_sc 已下载" if os.path.isdir(face) else "缺失")

    reid = os.path.join(PKG, "models", "reid", "osnet_x0_25_msmt17.pth")
    add(g, "ReID 人体特征权重", os.path.exists(reid),
        "%.1f MB" % (os.path.getsize(reid) / 1e6) if os.path.exists(reid) else "缺失")

    # Ollama
    code, out = run("timeout 8 curl -sS http://127.0.0.1:11434/api/version 2>&1")
    ollama_up = '"version"' in out
    add(g, "Ollama 服务", ollama_up, out.strip()[:50] if ollama_up else "未运行（先跑 start_ollama.sh）")
    if ollama_up:
        code, out = run("timeout 20 %s/.local/bin/ollama show qwen3-vl:2b-instruct 2>&1" % HOME)
        add(g, "视觉模型含 vision 能力", "vision" in out,
            "具备图片输入" if "vision" in out else "缺少 vision，动作识别会失效")


# ------------------------------------------------------------------ 3 核心模块
def check_modules():
    g = "核心代码模块"

    core = os.path.join(PKG, "scripts", "qwen_action_recognition_node.py")
    add(g, "动作识别核心文件存在", os.path.exists(core), core if os.path.exists(core) else "缺失")
    if not os.path.exists(core):
        return

    code, out = run_py('''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("c", %r)
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except Exception as exc:
    print(json.dumps({"error": str(exc)[:120]})); sys.exit(0)
need = ["PROMPT", "PoseActionAnalyzer", "PointCloudGroundAnalyzer",
        "parse_result", "merge_action_result"]
print(json.dumps({"missing": [n for n in need if not hasattr(m, n)]}))
''' % core)
    m = re.search(r'\{.*\}', out)
    if m:
        info = json.loads(m.group(0))
        if info.get("error"):
            add(g, "5 个必需符号齐全", False, "加载失败: " + info["error"])
        else:
            miss = info["missing"]
            add(g, "5 个必需符号齐全", not miss, "齐全" if not miss else "缺: " + ", ".join(miss))
    else:
        add(g, "5 个必需符号齐全", False, out.strip()[-80:])

    pose = os.path.join(HOME, "models", "yolo11n-pose.pt")
    code, out = run_py('''
import importlib.util
spec = importlib.util.spec_from_file_location("c", %r)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pa = m.PoseActionAnalyzer(%r, "cpu", 416, 0.25, 0.45, 4)
ok, msg = pa.initialize()
print("RESULT", ok, msg)
''' % (core, pose), env_extra={"FCHECK_CWD": PKG})
    add(g, "姿态分析器可加载", "RESULT True" in out,
        out.strip().splitlines()[-1][:70] if out.strip() else "")

    # 融合决策表（合成输入，不依赖硬件）
    code, out = run_py('''
import importlib.util
spec = importlib.util.spec_from_file_location("c", %r)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
G = {"status": "ground", "ground_like": True, "ground_confident": True}
E = {"status": "elevated", "ground_like": False}
U = {"status": "unknown"}
cases = [
    (("sitting", "chair", "unknown", U, False), ("sitting", "chair")),
    (("lying", "unknown", "unknown", G, True),  ("fallen", "floor")),
    (("lying", "unknown", "unknown", E, True),  ("lying", "unknown")),
    (("unknown", "unknown", "sudden_fall", U, True), ("sudden_fall", "floor")),
    (("unknown", "unknown", "waving", U, False), ("waving", "unknown")),
    (("waving", "unknown", "unknown", U, False), ("unknown", "unknown")),
]
bad = 0
for args, expect in cases:
    got = m.merge_action_result(*args)[:2]
    if got != expect:
        bad += 1
print("RESULT", bad)
''' % core)
    ok = "RESULT 0" in out
    add(g, "动作融合决策表 (6 例)", ok, "全部正确" if ok else out.strip()[-70:])

    add(g, "任务节点启动入口", os.path.exists(os.path.join(PKG, "scripts", "task1_find_owner_real.py")), "")
    add(g, "主人注册节点", os.path.exists(os.path.join(PKG, "scripts", "owner_voice_reid_test.py")), "")
    add(g, "一键启动脚本",
        os.path.exists(os.path.join(HOME, "catkin_ws", "competition", "robot.sh")), "")


# ------------------------------------------------------------------ 4 配置与资源
def check_config():
    g = "配置与资源"

    cfgp = os.path.join(PKG, "config", "task1_owner_search_real.yaml")
    code, out = run("python3 -c \"import yaml;d=yaml.safe_load(open('%s'));"
                    "print(d.get('action_llm_model'), d.get('action_llm_num_gpu'), d.get('electrical_switch_asr_engine'))\" 2>&1" % cfgp)
    add(g, "配置文件可解析", code == 0 and "qwen3-vl" in out, out.strip()[:60])

    core_path = os.path.join(PKG, "scripts", "qwen_action_recognition_node.py")
    add(g, "配置指向的核心文件存在", os.path.exists(core_path), "")

    pose_path = os.path.join(HOME, "models", "yolo11n-pose.pt")
    code, out = run("python3 -c \"import yaml;d=yaml.safe_load(open('%s'));print(d.get('action_model_path'))\" 2>&1" % cfgp)
    add(g, "姿态模型路径有效", out.strip() == pose_path, out.strip()[:70])

    maps = os.path.join(SRC, "wpb_home", "wpb_home_tutorials", "maps", "map.yaml")
    add(g, "地图文件就位", os.path.exists(maps), maps if os.path.exists(maps) else "缺失，导航不可用")

    wp = os.path.join(HOME, "waypoints.xml")
    add(g, "航点文件 (~/waypoints.xml)", os.path.exists(wp),
        "已存在" if os.path.exists(wp) else "未创建 —— 导航类功能不可用（可先用 navigate_enabled:=false 测其他功能）")

    owners = glob.glob(os.path.join(PKG, "data", "owner", "*.jpg")) + \
             glob.glob(os.path.join(PKG, "data", "owner", "*.jpeg"))
    add(g, "主人参考照片 (data/owner)", len(owners) > 0,
        "%d 张 —— 现场注册模式不需要照片" % len(owners) if owners else
        "0 张 —— 若用现场注册(owner_enrollment_enabled:=true)则无需照片")

    profiles = glob.glob(os.path.join(PKG, "data", "reid_owner_voice", "*.npz"))
    add(g, "已有注册档案", len(profiles) > 0,
        "%d 份" % len(profiles) if profiles else "无（首次需现场注册）")


# ------------------------------------------------------------------ 5 硬件
def check_hardware():
    g = "硬件设备"
    for dev, label in [("/dev/ftdi", "底盘串口"), ("/dev/rplidar", "激光雷达串口")]:
        ok = os.path.exists(dev)
        add(g, label + " " + dev, ok, "存在" if ok else "不存在（机器人没开机或没插好）")

    code, out = run("arecord -l 2>&1")
    add(g, "录音设备", "card" in out, (re.findall(r"card \d+: [^\[]+", out) or ["未发现"])[0][:50])
    code, out = run("aplay -l 2>&1")
    add(g, "播放设备", "card" in out, (re.findall(r"card \d+: [^\[]+", out) or ["未发现"])[0][:50])
    # USB 设备偶尔会重新枚举，多试几次，避免把“正在枚举”误报成“未识别”
    kinect_ok = False
    for _ in range(3):
        code, out = run("lsusb 2>&1 | grep -i '045e:02c4'")
        if code == 0:
            kinect_ok = True
            break
        time.sleep(2)
    add(g, "Kinect v2", kinect_ok, "已识别" if kinect_ok else "未识别（检查 USB 3.0 接口）")


# ------------------------------------------------------------------ 6 运行时
def check_runtime():
    g = "运行时（需栈已启动）"
    code, out = run("pgrep -x rosmaster", timeout=10)
    if code != 0:
        add(g, "ROS master", False, "未运行 → 其余运行时检查跳过")
        return
    add(g, "ROS master", True, "运行中")

    code, out = run("timeout 10 rosnode list 2>&1", timeout=14)
    nodes = [n for n in out.splitlines() if n.strip()]
    add(g, "节点数量", len(nodes) >= 8, "%d 个节点" % len(nodes))

    for topic, label, low in [
        ("/odom", "里程计", 50),
        ("/scan", "激光雷达", 5),
        ("/kinect2/qhd/image_color_rect", "Kinect 彩色图", 10),
        ("/perception/person_detections_2d", "行人检测", 3),
    ]:
        # Kinect 相关话题用“多次尝试”版本，避免设备休眠唤醒期间误报无数据
        if "kinect2" in topic or "perception" in topic:
            hz = topic_hz_patient(topic)
        else:
            hz = topic_hz(topic)
        add(g, label + " " + topic, bool(hz and hz >= low),
            ("%.1f Hz" % hz) if hz else "无数据")

    code, out = run("timeout 10 rostopic list 2>&1")
    add(g, "地图话题 /map", "/map" in out, "")
    add(g, "语音话题 /voice/say", "/voice/say" in out, "")


# ------------------------------------------------------------------ 7 需人工
def check_manual():
    g = "需要你操作确认"
    add(g, "身份识别 + 播报名字", None, "站到镜头前，应听到自己的名字")
    add(g, "动作：坐着", None, "坐下，应报“坐在…”")
    add(g, "动作：躺着", None, "躺下，应报“躺在…”")
    add(g, "动作：挥手", None, "挥手，应走近并问“需要什么帮助”")
    add(g, "动作：摔倒", None, "倒地，应报“摔倒”并伸出机械臂")
    add(g, "电器开关交互", None, "说“打开开关”，机械臂应抬到标记上方")
    add(g, "导航到航点", None, "需先建航点；能自主走到指定位置")
    add(g, "自主离场", None, "完成后导航到 exit 航点")
    add(g, "垃圾拾取", None, "尚未实现（赛项 300 分）")


# ------------------------------------------------------------------ 输出
def report():
    order = ["环境与依赖", "模型与权重", "核心代码模块", "配置与资源",
             "硬件设备", "运行时（需栈已启动）", "需要你操作确认"]
    groups = {}
    for g, n, s, d in results:
        groups.setdefault(g, []).append((n, s, d))

    total = len(results)
    npass = sum(1 for _, _, s, _ in results if s == "PASS")
    nfail = sum(1 for _, _, s, _ in results if s == "FAIL")
    nman = sum(1 for _, _, s, _ in results if s == "需人工")

    print()
    print("=" * 78)
    print("启智机器人比赛项目 · 功能体检报告")
    print("=" * 78)
    for g in order:
        if g not in groups:
            continue
        print()
        print("【%s】" % g)
        for n, s, d in groups[g]:
            mark = {"PASS": "\033[32m PASS \033[0m",
                    "FAIL": "\033[31m FAIL \033[0m",
                    "需人工": "\033[33m需人工\033[0m"}[s]
            print("  [%s] %-34s %s" % (mark, n, d))
    print()
    print("-" * 78)
    print("合计 %d 项：PASS %d ｜ FAIL %d ｜ 需人工 %d" % (total, npass, nfail, nman))
    print("-" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", action="store_true", help="额外检查正在运行的栈")
    args = ap.parse_args()

    print("正在体检，请稍候（静态检查约 1 分钟）...")
    check_environment()
    check_models()
    check_modules()
    check_config()
    check_hardware()
    if args.runtime:
        check_runtime()
    check_manual()
    report()


if __name__ == "__main__":
    main()
