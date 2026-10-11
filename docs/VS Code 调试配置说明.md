# VS Code 调试配置说明（launch.json / tasks.json / ros.env）

这份文档配套仓库里的 `vscode/` 目录：`launch.json`（25 个调试配置）、
`tasks.json`（一键起栈等任务）、`ros.env.example`（环境变量模板）。

---

## 1. 为什么不能直接用原来的 launch.json

原来的调试配置里写死了 56 处绝对路径，例如

```
/home/dyp/catkin_ws/src/wpb_service_robot/scripts/task_all_owners.py
```

这些路径只对"当时那台机器 + 那个用户名"有效，换一台电脑照抄过来，
VS Code 会直接报 `program ... does not exist`，一行都跑不起来。

所以仓库里的 `vscode/launch.json` 是**同一份配置的可移植版**：
所有的 `/home/dyp/catkin_ws` 换成了 VS Code 内置变量 `${workspaceFolder}`。

| 变量 | 展开成什么 |
| --- | --- |
| `${workspaceFolder}` | VS Code 当前打开的文件夹的绝对路径 |

因此使用时**必须直接用 VS Code 打开 `catkin_ws` 这个目录**
（文件 → 打开文件夹 → 选 `catkin_ws`），而不能打开它的上级目录或单个 `.py` 文件，
否则 `${workspaceFolder}` 会指错地方。

原来的 `/home/dyp/waypoints.xml` 也一起改成了
`${workspaceFolder}/src/wpb_service_robot/config/waypoints.xml`
（对应仓库里的 `ros_packages/wpb_service_robot/config/waypoints.xml`），
这样航点文件跟着代码走，不用再额外往家目录丢一份。

---

## 2. 怎么用

```bash
# 1) 仓库里的包放进工作区（首次）
cp -r ros_packages/* ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make

# 2) 调试配置放进 .vscode（注意目录名带点，是 .vscode）
cp vscode/launch.json      ~/catkin_ws/.vscode/launch.json
cp vscode/tasks.json       ~/catkin_ws/.vscode/tasks.json
cp vscode/ros.env.example  ~/catkin_ws/.vscode/ros.env

# 3) 修正 ros.env 里的路径（见下）
#    最省事的办法：VS Code 里 终端 → 运行任务 →「重新生成 ros.env」
```

三个文件的分工：

| 文件 | 作用 | 少了它会怎样 |
| --- | --- | --- |
| `launch.json` | 25 个调试配置（F5 直接跑某个脚本 / 起栈） | 没有调试配置可选 |
| `tasks.json` | 一键起栈、加载比赛参数、停止进程、重生成 ros.env | ①/⑪ 的 `preLaunchTask` 报"找不到任务" |
| `ros.env` | 注入 ROS 环境变量 | 节点 `ImportError`、找不到 `devel/lib` 里的库 |

---

## 3. 配置速查表

### 3.1 起栈（先起栈，再跑下面任何任务节点）

| 配置 | 类型 | 说明 |
| --- | --- | --- |
| ⑦【起栈】ROS: 启动机器人栈（含 TTS+ASR） | ros launch | 相机/雷达/Kinect/定位/导航/行人检测 + TTS + ASR。map 指向新包里的比赛地图。**平时用这个** |
| ⑦b【起栈】注册专用（ASR 窗口 2.5 秒 + 语速 0.85） | ros launch | 与 ⑦ 相同，只把语音采集窗口加长，报名字不容易被切碎。做注册（③b/③c/⑮）时用 |

> 起栈**不再预启动 Ollama**：动作识别已是纯 YOLO-Pose，用不到它，
> 留着的副作用是一堆代理告警 + 慢 6 秒。
> 例外：配置 ① / ⑪ 跑的是旧包节点 `task1_find_owner_real.py`，
> 里面还有 Qwen 电气开关意图识别，要跑这两个得先手动执行
> 任务「启动 Ollama（动作识别模型服务）」。

### 3.2 总任务（新包 wpb_service_robot，比赛用这套）

| 配置 | 前提 | 说明 |
| --- | --- | --- |
| ⑫ 新包·总任务 三人注册+认人+动作 | 起栈 + 定位初始化 | **比赛主配置**：注册 → 依次导航去 living_room/kitchen/bedroom/canteen 找人 → 报姓名 → 测动作 → 反应 → 回 exit。会真的开车 |
| ⑬ 新包·只测找人（跳过注册） | 起栈 + 定位初始化 | 与 ⑫ 相同但跳过注册，用已有档案。调找人逻辑时比 ⑫ 快 2 分钟 |
| ⑭ 新包·原地测找人（空跑不移动） | 起栈 | `_navigate_dry_run:=true`：当作"已经到客厅了"，原地转圈把找人→动作→反应跑完。只跑 living_room 一间 |
| ⑭b 原地测找人·GPU版 | 起栈 | 在 ⑭ 基础上把 ReID / 人脸 / 姿态挪到显卡（实测合计约 0.77GB）。做 CPU/GPU A/B 对照用 |
| ⑭c 原地测找人·快速扫描版（方案B） | 起栈 | 在 ⑭ 基础上加快转速、缩短停顿，一圈约 27 秒（原 41.5 秒） |
| ⑭d 摔倒靠近·导航版（方案B-2） | 起栈 + 定位初始化 | 摔倒后走近的方式换成走 move_base（绕开家具），失败逐级回退 |
| ⑮ 新包·注册+空跑去房间+真走近 | 起栈（⑦b） | 去房间不动，但挥手/摔倒时**真的**开车走近。现场调试这套正合适 |

关键参数（写在各配置的 `args` 里，改完重跑 F5 即可）：

| 参数 | 作用 |
| --- | --- |
| `_navigate_dry_run:=true` | 空跑：只挡"去航点"，不挡"走近主人" |
| `_approach_enabled:=true` | 走近主人总开关（挥手求助 / 摔倒协助） |
| `_approach_standoff_distance` | 走近后停在离主人多远（挥手 1.0m、摔倒 0.5m） |
| `_trash_announce_enabled:=false` | 垃圾识别已放弃（YOLO-World 的 CPU 占用设备撑不住），显式关掉 |
| `_action_pose_image_size:=640` | 姿态模型分辨率。默认 416，坐着挥手时手腕关键点常检不出来，⑮ 先提到 640 验证 |
| `_search_waypoints` / `_exit_waypoint_name` | 房间列表与出口航点名 |

### 3.3 旧包（wpb_task1_owner_search）单模块调试

| 配置 | 前提 | 说明 |
| --- | --- | --- |
| ①【节点】调试任务节点 task1_find_owner_real | 起栈（⑦b）+ Ollama | 旧的单体任务节点；带 `preLaunchTask` 自动把比赛参数加载到参数服务器 |
| ② 调试动作识别 qwen_action_recognition_node | 起栈 | 独立验证坐/躺/挥手/摔倒判定，按空格触发一次 |
| ③ 调试主人注册 owner_voice_reid_test | 起栈（⑦b） | 现场语音问名字 + 拍脸注册 1 人 |
| ③b 调试主人注册（2人） | 起栈（⑦b） | 依次注册 2 人 |
| ③c 调试人脸/身形分步注册（2人） | 起栈（⑦b） | 姓名加"确认"环节，人脸与身形分两步采集 |
| ③d 调试全流程 注册+寻找+动作 | 起栈（⑦b） | 一条龙：注册 → 旋转寻找 → 报动作 |
| ③e 只测找人（跳过注册） | 起栈 | 复用已有档案，跳过注册直接找人 |
| ③f 总任务 三人注册+逐位认人+动作 | 起栈（⑦b） | 旧包版总任务 |
| ③g 总任务·只测找人（跳过注册） | 起栈 | 旧包版，跳过注册 |
| ④ 调试人体 ReID person_reid_owner_test | 起栈 | 行人重识别单独验证 |
| ⑤ 调试行人检测 yoloworld_async | 起栈 | 需要 yoloworld_perception 的权重（不随仓库分发） |
| ⑨ 调试反馈节点 feedback_after_recognition | 起栈 + ⑩ | 收到动作结果后决定"说什么/做什么" |
| ⑩ 调试动作识别（静音版） | 起栈 | 与 ② 相同但 `_speak_enabled:=false`，避免和 ⑨ 重复播报 |
| ⑪ 调试接近主人（原地，不导航） | 起栈 + Ollama | 不导航，直接验证"找人 → 居中 → 识别 → 接近" |

### 3.4 其它

| 配置 | 说明 |
| --- | --- |
| ⑥ 全功能体检 feature_check.py | 纯 Python 检查，不需要 ROS master |
| ⑧ ROS: 附加到运行中的节点 | 节点已在终端跑着时，挂调试器看变量 |
| 【组合】动作识别⑩(静音) + 反馈⑨ | 一键同时起两个，看端到端效果 |

---

## 4. 典型使用流程

```
1. 终端执行任务「停止全部 ROS 进程」（等价 robot.sh stop，避免残留进程占着 Kinect）
2. 选 ⑦ 或 ⑦b 起栈，等 60~90 秒
3. 自检三条：
     rosnode list | grep -E "kinect2_stream_watchdog|offline_tts_node|offline_asr_node"
     rostopic hz /kinect2/qhd/image_color_rect          # 应 13~15Hz
     grep -oE "publishing rate: ~[0-9.]+Hz" ~/.ros/log/latest/rosout.log | tail -1
4. 跑任务：
     · 只调算法 → ⑭ / ⑭b / ⑭c（空跑，车不动）
     · 要验证走近 → ⑮
     · 完整比赛流程 → ⑫（先确认定位：rostopic echo -n1 /amcl_pose）
```

---

## 5. 常用路径对照表

仓库里的目录结构和实际工作区不是一对一，复制时按下表对应：

| 仓库内 | 放到本机 | 说明 |
| --- | --- | --- |
| `ros_packages/*` | `~/catkin_ws/src/*` | ROS 包，复制后要 `catkin_make` |
| `tools/*` | `~/catkin_ws/competition/*` | 一键启动脚本（tasks.json 里按 `competition/` 引用） |
| `vscode/*` | `~/catkin_ws/.vscode/*` | 本文件说的调试配置 |
| 厂商包 `wpb_home/` | 不在本仓库 | 厂商提供，`tasks.json` 里的「打开 RViz」用的是它的 `nav.rviz` |

> 如果不想放 `competition/`，把 `tasks.json` 里 4 处
> `${workspaceFolder}/competition/` 改成 `${workspaceFolder}/../tools/` 之类的实际位置即可。

---

## 6. 踩过的坑（照抄配置时最容易中的）

1. **ROS 私有参数会残留在 master 上。** 上一次运行传过的 `_xxx:=值`，下一次不写就沿用旧值。
   所以关键开关（`_navigate_enabled` / `_navigate_dry_run` / `_approach_enabled` /
   `_trash_announce_enabled`）在配置里都显式写死，复现问题时别删这些行。
2. **roslaunch 里没声明的 arg 会被静默忽略**，不报错。
   改起栈参数后要回终端确认真的生效（例如 `rosnode list` 里节点在不在）。
3. **改了 launch.json / tasks.json 要重新 F5**；改了 `.launch` 文件或 C++ 要重启栈；
   纯 Python 改动只需重跑任务节点。
4. **`envFile` 指向的文件不存在时，调试会直接起不来**，报错不明显。先确认
   `~/catkin_ws/.vscode/ros.env` 存在且 `PYTHONPATH` 里有 `devel/lib/python3/dist-packages`。
5. **⑦ 的 map 指向新包地图**（`ros_packages/wpb_service_robot/maps/map.yaml`）。
   用厂商教程地图导航必然失败（日志表现为 no global plan / 大绕路）。
   换地图只需改 ⑦/⑦b 的 `map:=` 参数，不用动代码。
6. **导航前必须初始化定位**，否则 move_base 一定规划不出来：
   RViz 里用 "2D Pose Estimate" 点机器人实际位置和朝向，再
   `rostopic echo -n1 /amcl_pose` 确认坐标对得上。
7. **⑫~⑮ 中除了 ⑭ 系列会把车真的开动**，请留出空间、手放急停。
8. **姿势模型跑在 CPU 上**（显存只有 4GB，要给别的模块留余量），
   别随手加 `cuda:0`；要加就用 ⑭b 那种单独做过显存核算的配置。

---

## 7. 没上传的 .vscode 文件

原来的 `.vscode/` 里还有 `settings.json`、`c_cpp_properties.json`、`extensions.json`，
这些是编辑器个人偏好（格式化规则、C++ 头文件搜索路径、推荐插件），
含大量本机路径且不影响 F5 调试，所以没有一起放进来。
需要的话从本机 `~/catkin_ws/.vscode/` 直接复制即可。
