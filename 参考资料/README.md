# 弱光特定对象识别及分级预警系统

当前实现覆盖 T02～T10 的代码级原型，并已具备 E0 离线实验回放链路：摄像头探测、实时预览、YuNet 人脸检测、人脸质量评估、授权对象注册、开放集身份匹配、IoU 多帧跟踪、区域进入/停留/离开状态机、本地 JSONL 事件日志和无窗口视频回放。程序只在本机读取视频流，注册时默认只保存 SFace 特征，不保存原始画面；真实摄像头、弱光阈值和连续运行验收仍待目标设备验证。

## 摄像头探测

默认探测索引 `0` 到 `4`：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml probe
```

Windows 下如果自动后端无法找到设备，可以分别尝试：

```powershell
.\.venv311\Scripts\python.exe run.py probe --backend dshow
.\.venv311\Scripts\python.exe run.py probe --backend msmf
```

## 实时预览

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml preview --index 0
```

按 `Q` 或 `Esc`，或关闭窗口即可退出。命令行参数可以临时覆盖摄像头配置：

```powershell
.\.venv311\Scripts\python.exe run.py preview --index 1 --width 640 --height 480 --fps 30 --no-mirror
```

长期使用的参数应写入 `config/default.yaml`。实际输出分辨率和帧率取决于摄像头驱动是否接受请求值。

项目同时提供标准 Python 包配置。需要使用 `lowlight-alert` 命令时，可以执行 `python -m pip install -e .`；日常开发直接使用 `run.py` 不需要安装。

## YuNet 人脸检测

启动实时人脸检测：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml detect --index 0
```

画面会显示人脸框、5 个关键点、检测置信度和当前人脸数量。检测阈值位于 `config/default.yaml` 的 `detection` 节；阈值越高，低置信度候选越容易被过滤。

此命令不会保存原始帧或人脸截图。当前 Windows 环境尚未发现可读摄像头，因此还需要在设备可用后完成真人正面及轻微侧脸的实机验收。

## 人脸质量评估

启动实时质量评估：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml quality --index 0
```

每张人脸会被检查以下指标：

- 人脸框最小尺寸。
- 亮度是否过暗或过曝。
- Laplacian 清晰度是否低于阈值。
- 眼睛、鼻尖和嘴角的相对位置是否表现出过大姿态。

绿色 `Quality OK` 表示当前帧可进入后续注册或识别，橙色框会显示 `small`、`dark`、`bright`、`blur`、`pose` 或 `crop` 等拒绝原因。阈值位于 `config/default.yaml` 的 `quality` 节，当前数值是工程初值，必须在摄像头可用后用实拍画面标定。

## 授权对象注册

摄像头可用后，为一名已获得明确同意的对象采集模板：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml enroll --subject-id person-a --display-name "Person A" --count 20
```

`subject-id` 是稳定的本地编号，只能包含英文字母、数字、连字符和下划线。建议使用 `person-a` 这类非敏感编号，不要使用身份证号、手机号或学号。

注册流程只接受画面中恰好一张脸、质量合格、满足采样间隔且不是近重复帧的样本。每个合格模板会立即原子写入 `data/templates/<subject-id>.npz`；中途退出时已写入的模板会保留，原始帧不会保存。

查看已注册对象和模板数量：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml subjects
```

模板文件虽然不含原始照片，但 SFace 特征仍属于敏感生物识别数据。`data/` 已被 Git 忽略；当前原型没有实现模板加密，应依靠本机账户和目录权限保护，并按参与者同意的期限删除。

## 实时身份匹配

已完成注册对象后，可以启动开放集匹配预览：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml recognize --index 0
```

系统只对质量合格的人脸进行匹配，并显示 `Registered`、`Unknown` 或 `Uncertain`。接受阈值、拒绝阈值和灰区间隔位于 `config/default.yaml` 的 `recognition` 节；当前值是普通光照基线初值，不能替代项目数据标定。没有模板时命令仍可运行，但所有合格人脸会显示为 `Unknown`。

## 本地监控与事件日志

`monitor` 将检测、质量门控、开放集识别、短时跟踪和区域状态机串成一条本地闭环。它只对已确认轨迹产生区域事件，并把每个事件追加到 JSONL；同一 `event_id` 重复写入时会被幂等过滤。

先在 `config/default.yaml` 的 `events.zones` 配置至少一个归一化多边形。坐标是相对于画面宽高的 `[0, 1]` 值，例如：

```yaml
events:
  log_path: logs/events.jsonl
  confirm_frames: 3
  max_missing_frames: 5
  lost_tolerance_seconds: 1.0
  dwell_seconds: 2.0
  cooldown_seconds: 10.0
  zones:
    - name: door-east
      severity: 2
      polygon: [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]
```

启动本地监控：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml monitor --index 0
```

窗口中的短时丢帧会保留原轨迹，超过 `lost_tolerance_seconds` 才产生离开事件；跟踪器确认轨迹结束时会立即写入离开事件。日志包含事件类型、区域、轨迹、身份状态、相似度、等级、原因和证据标记。当前原型的 `observed_at`/`first_seen` 是用于时长计算的数值时钟，外部审计展示前还应补充带时区的 wall-clock 字段。默认配置的 `zones` 为空，未配置区域时命令会明确报错，不会静默监控整个画面。

当前只实现本地记录和画面提示，未接入 MQTT/HTTPS、声音或手机通知；原始视频和抓拍也不会由此命令自动保存。

## 离线基线评测

将已采集的特征比较结果整理为 CSV（至少包含 `genuine,score`，可选 `condition`），例如：

```text
genuine,score,condition
true,0.91,normal
false,0.22,dark
```

使用配置中的接受阈值评测：

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml evaluate data/scores.csv
```

也可以临时指定阈值。命令会输出总体和各条件的 TAR、FMR、FNMR；CSV 中的分数必须是余弦相似度，范围为 `-1` 到 `1`。阈值仍需在独立测试集上冻结，不能把公开模型示例阈值直接当作最终结果。

## 离线视频回放（E0）

回放命令在无摄像头、无 GUI 环境下读取已保存的视频，并复用 `monitor` 的检测、质量、跟踪和区域事件链路。配置必须包含至少一个 `events.zones`；建议复制并修改 `config/default.yaml`，不要直接使用其中的空区域配置。

```powershell
.\.venv311\Scripts\python.exe run.py --config config/default.yaml replay `
  data\experiments\exp-lowlight-001\sources\run-normal-001.mp4 `
  --condition normal `
  --experiment-id exp-lowlight-001-run-normal-001
```

四类条件只能使用 `normal`、`dim`、`backlight`、`near_black`。回放默认读取视频 FPS，并以 `frame_index / fps` 生成 `source_time_s`；FPS 元数据无效时必须显式指定 `--source-fps`。`--no-mirror` 用于关闭配置中的镜像，镜像策略必须同步记录到实验 manifest；`--max-frames` 只适合调试，会把报告标记为 `termination_reason=max_frames` 和 `source_complete=false`。

每个 `experiment-id` 都会在 `logs/replays/<experiment-id>/` 建立不可覆盖的独立目录，产物为：

```text
events.jsonl   # 事件日志，含 reason；收尾 machine_flush 单独标记
frames.csv     # 逐帧遥测，时间列为 source_time_s
report.json    # 输入/模型/配置哈希、终止原因、计数和产物元数据
```

回放会先写临时产物，全部校验通过后再发布；只有 `report.json` 存在且 `status` 为 `complete` 才算完整 run。坏视频、无效 FPS、声明帧数不完整、尺寸变化或重复输出目录会明确失败。报告中的 `detections`、`registered`、`unknown`、`uncertain` 和事件数是按帧的系统观察量，不是准确率、人数或真实误报率；只有配合人工真值 `observations.csv` 和 `event_truth.csv` 才能计算召回率、FMR/FNMR 或事件 F1。视频 EOF 触发的会话收尾事件不应直接当作真实离开事件。

## 开发检查

```powershell
.\.venv311\Scripts\python.exe -m ruff check .
.\.venv311\Scripts\python.exe -m pytest
```

运行期间产生的人脸样本、特征模板、抓拍和日志必须保存在已忽略的 `data/`、`captures/`、`logs/` 目录，不应提交到 Git。

## 项目规划与知识资料

- [项目会后共识与下一阶段路线](项目会后共识与下一阶段路线.md)：当前共识、阶段目标、验收门和工作包接口。
- [弱光特定对象识别与分级预警知识手册](弱光特定对象识别与分级预警知识手册.md)：弱光成像、YuNet/SFace、阈值评测、多帧事件、边缘部署、通信和隐私安全参考。
- [弱光实验执行与标注规范](弱光实验执行与标注规范.md)：E0/E1 的实验目录、manifest、逐帧标注、事件真值、指标口径和授权/删除要求。
