# yoloworld_perception（本机适配版）

这是为 `wpb_task1_owner_search` 补的**模型资源包**，不是上游完整的
`yoloworld_perception`。

## 为什么需要它

上游的 launch 文件里有三处这么写：

```xml
<param name="model_path" value="$(find yoloworld_perception)/models/yolov8s-world-person.pt" />
```

`$(find ...)` 在 roslaunch 解析阶段就会执行，包不存在会直接报
`package not found` 并终止启动。但上游的**推理脚本**
（`yoloworld_async.py`、`yoloworld_debug_viewer_stable.py`）
已经内置在 `wpb_task1_owner_search` 包里了，所以这里唯一需要提供的
就是那个 `.pt` 权重文件。

## 内容

```
models/yolov8s-world-person.pt   # 由 ~/models/yolov8s-world.pt 改名而来
```

改名只是因为上游写死了 `-person` 这个文件名。launch 里同时设了
`classes=person`，所以通用版 YOLO-World 权重可以直接当行人检测模型用。

## 如果要换成真正的行人专用权重

把下载到的权重覆盖到 `models/yolov8s-world-person.pt` 即可，路径不用改。
