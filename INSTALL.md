# 部署与依赖安装

目标机器：启智 ROS 机器人的机载电脑（Ubuntu 20.04 + ROS Noetic）。

---

## 0. 前置：ROS 工作区

确认已有 catkin 工作区，并且能正常 source：

```bash
ls ~/catkin_ws/src
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
echo $ROS_DISTRO        # 应为 noetic
```

---

## 1. 第三方 ROS 包（本仓库不含）

以下包需要另外获取并放到 `~/catkin_ws/src/`，它们都是开源项目：

| 包 | 用途 | 来源 |
|---|---|---|
| `wpb_home` | 底盘驱动、教程、本地规划器 | 启智官方（随机器人附带的源码包） |
| `wpr_simulation` | 仿真与部分模型资源 | 启智官方 |
| `iai_kinect2` | Kinect v2 驱动 | github.com/code-iai/iai_kinect2 |
| `waterplus_map_tools` | 航点管理（MapTools 插件） | 启智官方 |

> 如果机器人原本就能跑 `roslaunch wpb_home_bringup minimal.launch`，
> 说明这些包已经就位，跳过本节。

---

## 2. 放入本仓库的 4 个包并编译

```bash
cp -r ros_packages/* ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

验证：

```bash
for p in wpb_task1_owner_search offline_voice_bridge perception_msgs yoloworld_perception; do
  echo -n "$p -> "; rospack find $p
done
```

四个都应打印出路径。若有 `MISSING`，说明该包没复制进去或编译失败。

---

## 3. Python 依赖

### 3.1 常规安装（能连外网时）

```bash
python3 -m pip install --user \
  funasr modelscope sherpa-onnx sounddevice \
  insightface onnxruntime \
  ultralytics opencv-python \
  yacs future termcolor gdown h5py tensorboard zstandard
```

### 3.2 离线安装（网络受限时）

本机经验：`pypi.org`、`github.com` 可能不通，但 `codeload.github.com` 与
`files.pythonhosted.org` 通常可达。纯 Python 包可以直接把目录放进
`site-packages`，不必走 pip：

```bash
SP=~/.local/lib/python3.8/site-packages
cd /tmp && mkdir -p dep && cd dep
curl -sSL -o ftfy.zip "https://codeload.github.com/rspeer/python-ftfy/zip/refs/tags/v6.0.1"
python3 -c "import zipfile; zipfile.ZipFile('ftfy.zip').extractall('.')"
cp -r python-ftfy-6.0.1/ftfy "$SP/"
```

> **版本注意**：`ftfy` 最新版要求 Python ≥ 3.9，本机是 3.8，**必须用 v6.0.1**。
> 用 main 分支会在 import 时报 `TypeError: 'type' object is not subscriptable`。

### 3.3 YOLO-World 必需的 CLIP

`ultralytics` 的 YOLO-World 需要 CLIP 文本编码器，缺了会让行人检测**静默失效**
（节点进程存在，但 `/perception/person_detections_2d` 永远没有数据）：

```bash
python3 -m pip install --user zstandard
U="https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"
curl -sSL -o /tmp/clip.zip "$U"
cd /tmp && python3 -c "import zipfile; zipfile.ZipFile('clip.zip').extractall('.')"
cp -r CLIP-main/clip ~/.local/lib/python3.8/site-packages/
```

验证：

```bash
python3 -c "import clip, ftfy, wcwidth; print('CLIP OK', ftfy.__version__)"
```

> CLIP 的模型权重（ViT-B-32，约 337 MB）首次运行 YOLO 时会自动下载到
> `~/weights/clip/ViT-B-32.pt`。**首次启动会等 3-4 分钟**，之后就快了（约 5 秒）。

### 3.4 人体 ReID（torchreid）

源码已随本仓库提供（`wpb_task1_owner_search/third_party/deep-person-reid`），
**不需要 `pip install -e`**，脚本会自己把它加进 `sys.path`，只缺依赖：

```bash
python3 -m pip install --user yacs future termcolor gdown h5py tensorboard
```

验证：

```bash
cd ~/catkin_ws/src/wpb_task1_owner_search
python3 -c "
import sys; sys.path.insert(0, 'third_party/deep-person-reid')
import torchreid; print('torchreid', torchreid.__version__)
"
```

> **ReID 专用权重需要自己放一份**：`<包>/models/reid/osnet_x0_25_msmt17.pth`
> （`wpb_task1_owner_search` 和 `wpb_service_robot` 各需要一份）。
> 它体积较大（约 9MB），没有随仓库提交，获取方式二选一：
>
> 1. 从本仓库的历史提交里取（该文件曾经提交过）：
>    ```
>    git show e7ce74f:ros_packages/wpb_task1_owner_search/models/reid/osnet_x0_25_msmt17.pth \
>      > ros_packages/wpb_task1_owner_search/models/reid/osnet_x0_25_msmt17.pth
>    ```
> 2. 用 torchreid 自己下载：它是 `osnet_x0_25` 在 MSMT17 上的预训练权重，
>    由 torchreid 的 `download_model('osnet_x0_25', 'msmt17')` 取得，改名后放到上面的路径。
>
> 缺少这个权重时，程序会回退到 ImageNet 预训练权重，**认人的区分度会明显变差**
> （实测两位不同的人相似度从 0.57 涨到 0.75），所以务必放对。

---

## 4. Ollama（动作识别用的视觉模型）

### 4.1 安装 Ollama（免 sudo）

官方二进制是 `.tar.zst`，本机没有 `zstd`，用 Python 解压：

```bash
python3 -m pip install --user zstandard
U="https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-linux-amd64.tar.zst"
curl -sSL -o /tmp/ollama.tar.zst "$U"     # 约 1.4 GB

python3 - <<'PY'
import zstandard, tarfile, os
with open('/tmp/ollama.tar.zst','rb') as f:
    with zstandard.ZstdDecompressor().stream_reader(f) as r:
        tarfile.open(fileobj=r, mode='r|').extractall(os.path.expanduser('~/.local'))
PY

~/.local/bin/ollama --version      # 应输出 0.34.4
```

### 4.2 安装视觉模型

**关键：不要用 `ollama pull`**。它走代理时会报 `EOF`，直连又很慢
（模型主体在 Cloudflare R2 上）。用仓库里的脚本：

```bash
bash tools/start_ollama.sh                                              # 启动服务
python3 tools/pull_ollama_model.py qwen3-vl 2b-instruct                 # 手动装模型
ollama list
ollama show qwen3-vl:2b-instruct        # Capabilities 里必须有 vision
```

> **必须确认有 `vision`**。上游作者配置里写的 `qwen3.5:0.8b` 是**纯文本模型**，
> 不支持图片输入，动作识别会完全失效。目前已改为 `qwen3-vl:2b-instruct`。

### 4.3 性能参考（GTX 1650 4GB）

| 运行方式 | 冷启动 | 热推理 |
|---|---:|---:|
| CPU（`action_llm_num_gpu: 0`，默认） | 22.4s | 19.2s |
| GPU / Vulkan（`-1`） | 12.0s | 8.0s |

驱动 535 对 Ollama 来说偏旧（要求 550+），它会打印
`NVIDIA driver too old` 并自动退回 **Vulkan** 后端 —— 功能正常。

---

## 5. 地图与航点

### 5.1 地图

launch 默认读 `$(find wpb_home_tutorials)/maps/map.yaml`，
但该目录被 `.gitignore` 忽略、默认是空的，需要手动放：

```bash
cp <你的地图>/map.pgm <你的地图>/map.yaml \
   ~/catkin_ws/src/wpb_home/wpb_home_tutorials/maps/
```

验证：

```bash
cd ~/catkin_ws && source devel/setup.bash
rosrun map_server map_server ~/catkin_ws/src/wpb_home/wpb_home_tutorials/maps/map.yaml
# 另开终端：rostopic echo -n1 /map | head
```

### 5.2 航点

航点文件路径是 `~/waypoints.xml`（launch 默认值）。
**文件不存在时 `wp_manager` 只打一条日志、航点表为空，所有按名字导航都会失败。**

建航点步骤：

1. 启动 `bash tools/robot.sh stack`（需要 RViz，用 `SHOW_RVIZ=1`）
2. RViz 里用 **`2D Pose Estimate`** 给机器人初始位姿
3. 遥控机器人到目标位置：`rosrun teleop_twist_keyboard teleop_twist_keyboard.py`
4. RViz **工具栏**里的 **`AddWaypoint`** 工具 → 点地图上机器人位置 → 输入名字
5. 保存：`rosrun waterplus_map_tools wp_saver`（默认存到 `~/waypoints.xml`）

本赛项需要的航点名：`living_room`、`kitchen`、`bedroom`、`canteen`、`exit`。

> 航点坐标与地图**强绑定**：换地图后航点全部失效，必须重采。

---

## 6. 主人参考数据

本仓库**不包含**任何人的照片。你有两种方式：

| 方式 | 做法 |
|---|---|
| 照片（默认模式） | 把 3 位主人的多角度单人照放进 `wpb_task1_owner_search/data/owner/`（正面/左/右/半侧），然后**不要**开 `owner_enrollment_enabled` |
| 运行期注册 | 启动时加 `owner_enrollment_enabled:=true`，机器人会语音问名字并现场采脸（**只在内存，重启需重做**） |

查看注册结果时务必确认 `has_face = True`（缺人脸特征等于白注册）：

```bash
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob('/home/*/catkin_ws/src/wpb_task1_owner_search/data/reid_owner_voice/*.json')):
    m = json.load(open(f))
    print(f.split('/')[-1], '| name =', m.get('owner_name'),
          '| has_face =', m.get('has_face_embedding'))
PY
```

---

## 7. 验证部署是否成功

```bash
# 硬件自检
bash tools/robot_selfcheck.sh

# 启动核心服务并查看状态
bash tools/start_ollama.sh
bash tools/robot.sh stack      # Ctrl+C 停止
bash tools/robot.sh status     # 另开终端看状态
```

`status` 应该看到 `/odom`（约 100Hz）、`/scan`（约 12Hz）、
`/kinect2/qhd/image_color_rect`（约 30Hz）、
`/perception/person_detections_2d`（约 14Hz）都有数据。

---

## 8. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `catkin_create_pkg: error: ... required: --rosdistro` | 没 source ROS。先 `source /opt/ros/noetic/setup.bash` |
| 命令行传的参数没生效 | roslaunch 不会为未声明的 arg 报错，**参数被静默忽略**。先确认目标 launch 是否声明了它 |
| `Map_server could not open .../map.yaml` | `wpb_home_tutorials/maps/` 是空目录，见第 5.1 节 |
| `Failed to load waypoints` | 没有 `~/waypoints.xml`，见第 5.2 节 |
| `/perception/person_detections_2d` 无数据 | 缺 CLIP 依赖，见第 3.3 节；看 `yoloworld-*.log` 是否出现 `warm-up complete` |
| 动作识别完全不工作 | 缺 `qwen_action_recognition_node.py`，或任务日志里没有 `Owner Qwen action recognition ready` |
| 音箱没声音 | 本机 ALSA 默认设备指向 HDMI，需指定 `plughw:CARD=Generic_1,DEV=0` |
| Kinect 丢包 | 必须独占 USB 3.0 口，不要和其他大带宽设备共用一个 hub |
| 机器人认错人 | `data/reid_owner_voice/` 里可能是别人的档案，删掉重新注册 |
