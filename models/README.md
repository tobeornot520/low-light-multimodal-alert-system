# Model files

Model binaries are intentionally excluded from Git and team source archives.

Download the OpenCV Zoo models into this directory:

```text
face_detection_yunet_2023mar.onnx
face_recognition_sface_2021dec.onnx
```

Official sources:

- YuNet: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- SFace: <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>

Expected SHA-256 values for the files used by the 2026-08-29 snapshot:

```text
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  face_detection_yunet_2023mar.onnx
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  face_recognition_sface_2021dec.onnx
```

PowerShell verification:

```powershell
Get-FileHash -Algorithm SHA256 models\face_detection_yunet_2023mar.onnx
Get-FileHash -Algorithm SHA256 models\face_recognition_sface_2021dec.onnx
```

Linux verification:

```bash
sha256sum models/*.onnx
```

Do not continue when a hash differs. Re-download the model from an approved source.
Review each upstream model card and license before downloading or redistributing a model.
