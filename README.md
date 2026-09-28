# 启智机器人 · 居家生活机器人赛项（2026）

本仓库是队伍的**比赛工作仓库**，包含为参赛而修改/新增的 ROS 包、启动脚本和备赛文档。

> 上游的 `wpb_home`、`wpr_simulation`、`iai_kinect2`、`waterplus_map_tools` 等第三方包
> **不在本仓库内**，需要按 [INSTALL.md](INSTALL.md) 单独获取。

---

## 目录结构

```
├── ros_packages/                 要放进 catkin_ws/src 的 ROS 包
│   ├── wpb_task1_owner_search/   主任务包（改动最大：任务状态机、动作识别、语音）
│   ├── offline_voice_bridge/     离线语音（Piper 中文 TTS + SenseVoice ASR）
│   ├── perception_msgs/          自定义消息（Detection2D / Detection2DArray）
│   └── yoloworld_perception/     YOLO-World 行人检测的模型资源包
├── tools/                        一键启动与辅助脚本
│   ├── robot.sh                  统一入口：栈/注册/任务/停止/状态
│   ├── start_ollama.sh           启动本地视觉模型服务
│   ├── robot_selfcheck.sh        硬件自检（底盘/雷达/相机/声卡/GPU）
│   └── pull_ollama_model.py      绕开 ollama 下载器装模型
└── docs/                         备赛文档
    ├── 备赛方案.md               总体方案 + 比赛规则拆解 + 已踩的坑
    ├── 17天冲刺计划.md           倒排日程
    └── 多主人支持改造方案.md      未完成项的改造评估
```

---

## 快速开始

完整步骤见 [INSTALL.md](INSTALL.md)，这里是最短路径：

```bash
# 1) 把 4 个包放进工作区
cp -r ros_packages/* ~/catkin_ws/src/

# 2) 编译
cd ~/catkin_ws && catkin_make && source devel/setup.bash

# 3) 自检硬件
bash tools/robot_selfcheck.sh

# 4) 一键启动（栈 + 主人注册，单人测试）
bash tools/robot.sh enroll
```

其他启动方式：

| 命令 | 作用 |
|---|---|
| `bash tools/robot.sh stack` | 只启动机器人栈（相机/雷达/YOLO/语音/导航） |
| `bash tools/robot.sh enroll` | 栈 + 主人注册（默认注册 1 人） |
| `bash tools/robot.sh task` | 栈 + 任务节点（默认不导航） |
| `OWNER_COUNT=3 bash tools/robot.sh enroll` | 注册 3 位主人 |
| `NAVIGATE=1 SHOW_RVIZ=1 bash tools/robot.sh task` | 带导航和 RViz 跑任务 |
| `bash tools/robot.sh stop` | 停掉所有相关进程 |
| `bash tools/robot.sh status` | 查看节点/话题/服务状态 |

---

## 系统架构（脚本之间的调用关系）

```
tools/robot.sh
  ├─ start_ollama.sh           -> Ollama(127.0.0.1:11434) + qwen3-vl:2b-instruct
  └─ roslaunch wpb_task1_owner_search ...
       ├─ task1_owner_search_bringup.launch        机器人栈
       │    ├─ wpb_real_navigation.launch           底盘/雷达/Kinect/地图/定位/导航
       │    ├─ yoloworld_async.py                   行人检测 -> /perception/person_detections_2d
       │    └─ offline_voice_bridge                 Piper TTS + SenseVoice ASR
       ├─ task1_owner_search_task_only.launch       任务节点（调试用）
       │    └─ task1_find_owner_real.py             ← 任务状态机（核心）
       │         ├─ 动态加载 offline_voice_bridge/scripts/qwen_action_recognition_node.py
       │         │     提供 PoseActionAnalyzer / PointCloudGroundAnalyzer
       │         │          PROMPT / parse_result / merge_action_result
       │         ├─ HTTP -> Ollama(qwen3-vl:2b-instruct)   动作判定（视觉）
       │         ├─ InsightFace -> 人脸身份
       │         └─ /wpb_home/mani_ctrl -> 机械臂
       └─ owner_voice_reid_test.launch              主人注册（语音报名字 + 人体 ReID）
```

---

## 环境要求

| 项目 | 版本 |
|---|---|
| 系统 | Ubuntu 20.04 |
| ROS | Noetic |
| Python | 3.8 |
| GPU | GTX 1650 4GB（驱动 535，Ollama 走 Vulkan 后端） |
| 机器人 | 启智 ROS 机器人（`/dev/ftdi` 底盘、`/dev/rplidar` 雷达、Kinect v2） |

---

## 已知限制

| 项 | 状态 |
|---|---|
| 多主人（3 位）支持 | **未实现**，任务节点目前只认 1 个身份，评估见 `docs/多主人支持改造方案.md` |
| 垃圾拾取 | **未实现**（赛项 300 分） |
| 比赛场地地图 / 航点 | 需现场采集，仓库内的地图是练习场地图 |
| `data/owner/` 参考照片 | **已清空**，需放入自己队伍主人的照片 |
| 动作识别提示词 | 本地补写，未经真实"坐/躺/摔倒"场景的准确率验证 |

---

## 重要说明

- 本仓库**不包含**任何人的照片、人脸特征或生物特征档案，请勿把注册档案提交上来。
- 两个模型权重（`osnet_x0_25_msmt17.pth` 9MB、`yolov8s-world-person.pt` 26MB）
  已随仓库提供，开箱可用。
- 大模型（SenseVoice、Ollama 的 qwen3-vl）体积较大，不随仓库分发，
  安装方法见 [INSTALL.md](INSTALL.md)。
