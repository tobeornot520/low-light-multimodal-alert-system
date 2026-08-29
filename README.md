# 弱光特定对象识别及分级预警系统

面向固定、授权场景的研究原型：在弱光条件下对已登记对象进行开放集识别，结合多帧轨迹和区域行为形成可解释的分级事件，并为 RGB/NIR 多模态融合和边缘部署建立可复现实验基础。

## 当前状态

当前可运行链路以单路 RGB 为主，已实现：

- 摄像头探测、预览和 YuNet 人脸检测；
- 输入质量门控、SFace 模板注册和开放集身份匹配；
- 多帧跟踪、身份投票以及区域进入、停留、离开事件；
- 本地 JSONL 日志、无窗口视频回放和逐帧审计报告；
- 实验 manifest 校验、身份与事件评测；
- RGB/NIR 数据协议和保守融合内核。

NIR 实机适配、同步双模数据、目标设备阈值、行为早期识别模型、可靠远程通知和边缘设备验收尚未完成。当前结果不能视为面向任意人员、任意环境的通用识别系统，也不能用于自动执法或无人复核的高影响决策。

## 环境安装

建议使用 Python 3.11。

```bash
git clone https://github.com/tobeornot520/low-light-multimodal-alert-system.git
cd low-light-multimodal-alert-system
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-dev.txt
```

Windows PowerShell：

```powershell
git clone https://github.com/tobeornot520/low-light-multimodal-alert-system.git
cd low-light-multimodal-alert-system
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv311\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## 模型文件

模型二进制不进入 Git。按照 [models/README.md](models/README.md) 从 OpenCV Zoo 下载并校验以下文件：

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

没有模型和摄像头时仍可运行自动化测试及 manifest 校验。

## 快速验证

以下命令均从仓库根目录执行：

```bash
python run.py --help
python run.py validate-manifest config/experiment.example.yaml
python run.py validate-manifest config/multimodal-manifest.example.yaml
python -m pytest
python -m ruff check .
```

摄像头和模型准备好后，可以逐步检查实时链路：

```bash
python run.py --config config/default.yaml probe
python run.py --config config/default.yaml detect --index 0
python run.py --config config/default.yaml quality --index 0
python run.py --config config/default.yaml recognize --index 0
```

注册已明确同意的参与者：

```bash
python run.py --config config/default.yaml enroll \
  --subject-id person-a --display-name "Person A" --count 20
```

`monitor` 和 `replay` 要求配置至少一个 `events.zones`。默认配置有意保持为空，避免未明确区域时启动监测。完整命令与输出说明见 [运行说明](参考资料/README.md)。

## 项目资料

- [运行说明](参考资料/README.md)
- [组员同步与上手说明](参考资料/组员同步与上手说明.md)
- [项目会后共识与下一阶段路线](参考资料/项目会后共识与下一阶段路线.md)
- [弱光实验执行与标注规范](参考资料/弱光实验执行与标注规范.md)
- [知识手册](参考资料/弱光特定对象识别与分级预警知识手册.md)
- [项目申报书准备稿](参考资料/项目申报书准备稿.md)

提交代码前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 数据与隐私边界

- 只采集明确同意的参与者，并使用 `person-a` 一类项目内编号；
- 原始视频、照片、模板、标注、同意材料和运行日志不得提交到 Git；
- `data/`、`captures/`、`logs/`、`.env` 和 `models/*.onnx` 已被忽略；
- 注册、开发和测试按人员及采集会话隔离，冻结测试集不参与调参；
- 身份与行为分别推断，身份本身不直接决定风险等级；
- 对外演示、数据传输和长期保留前，必须完成学校要求的伦理与数据审批。

本仓库当前未声明开源许可证。公开可见不等于授予复制、修改或再发布许可；团队如需对外开源，应先共同确认许可证和素材权利。
