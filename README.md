# 目标检测模型评估与分析软件 V1.1

将 Ultralytics YOLO 目标检测 `.pt` 模型和带标注的数据集放入本地，通过桌面窗口或命令行评估。类别来自模型，不绑定具体行业。

## 安装与启动

建议使用 Python 3.11 或 3.12 的独立环境，并安装带 Tk 的标准 Python。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

已安装依赖的环境可直接运行 `python app.py`，或双击 `启动软件.cmd`。启动入口优先使用项目内 .venv，其次尝试本机 miniconda3 的 pytorch 环境，最后使用默认 Python。如需 GPU，请按 PyTorch 官方安装说明选择与显卡驱动适配的版本，设备栏填写 `0`；默认 `cpu` 不需要显卡。窗口中显示缺少依赖时，应在启动窗口所用的同一 Python 环境中安装。

## 数据准备

推荐结构：

```text
dataset/
  images/test/a.jpg
  labels/test/a.txt
```

也支持 `dataset/test/images` 配对 `dataset/test/labels`，以及 `dataset/images` 配对 `dataset/labels`。子目录按相对路径配对。每行标签为：

```text
0 0.5 0.5 0.2 0.3
```

五列分别是类别编号、中心横坐标、中心纵坐标、宽、高；坐标归一化为 0～1，编号与模型一致。无目标图片请提供同名空 `.txt`，本软件采用严格配对检查，避免漏标被误认为负样本。整个测试集至少包含一个真实目标。标签框必须位于图片内。

若不提供 data.yaml，类别直接取自模型，请自行确认测试集编号含义正确。也可选择 data.yaml（优先于根目录），例如：

```yaml
path: ./dataset
test: images/test
names:
  0: person
  1: car
```

相对 path 以 YAML 所在目录为基准；test/val 路径相对于 path。类别名称和编号必须与模型完全一致。第一版仅支持目录型划分，暂不支持图片清单 txt、多个目录列表、COCO JSON 或 VOC XML。

## 操作步骤

1. 选择自己有权使用、来源可信的 `.pt` 检测模型。
2. 选择带标签的数据集根目录，或选择 data.yaml。
3. 选择 test 或 val 划分、输出位置和参数，然后点击“开始评估”。
4. 日志显示数据检查及评估进度；需要中止时点击“停止评估”。
5. 完成后点击“查看报告与错例”，浏览指标和曲线。打开“查看与筛选错例”，按错误类型、类别筛选，点击图片名称查看叠加框及明细。
6. 报告自动导出至每次任务的独立目录；可复制整个目录分享，或用浏览器打印为 PDF。

## 结果文件

- `report.html`：离线网页报告，引用同目录下的结果图；分享时请一起复制整个任务目录。
- `metrics.json`：总体、各类别指标、数据统计、参数及模型 SHA256。
- `metrics.csv`：可在表格软件中打开的指标表。
- `metrics.txt`、`map.txt`：文本摘要与达标状态。
- `resolved_data.yaml`：本次实际使用的绝对路径配置。
- `test_result/`：Ultralytics 生成的曲线、混淆矩阵和批次样例预测图，样例不保证覆盖全部图片。
- `image_predictions.json`：本次验证中全部图片的真实框和经过模型后处理的预测框，包含置信度和原图像素坐标。无预测的图片也有记录。
- `diagnostics.html`、`diagnostics/`：可离线浏览的错例列表、逐图明细和预览。复制整个任务目录时，预览会随报告一起复制。
- `diagnostics.csv`：整个测试集的错例清单，每行一个错例，可按图中 G（真实框）和 P（预测框）编号复查。

## 逐图错例口径

“错例置信度”留空时取 0.25 和评估置信度下限中的较大值，也可以手动填写。命令行使用 `--diagnostic-conf 0.25`。只有置信度达到错例阈值的预测才参与诊断，匹配 IoU 固定为 0.50。

匹配时先按 IoU 从高到低配对同类框，再配对剩下的异类框。每个框最多匹配一次，IoU 相同时优先置信度较高的预测。异类配对记为一次类别混淆，剩余未匹配的真实框记为漏检，剩余未匹配的预测框记为误检。主报告中的 mAP 等验收指标仍由 Ultralytics 计算。

无真实目标的类别指标显示“未参与评估”，不会用总体指标代替。达到阈值不代表模型可用于任何实际场景；结果取决于测试集的标注与代表性。

## 命令行

```powershell
python evaluate_model.py --weights D:/models/best.pt --input-dir D:/dataset --split test --device cpu --output-dir ./results
python evaluate_model.py --weights D:/models/best.pt --data D:/dataset.yaml --split val --device cpu
```

退出码：0 表示评估完成且 mAP50 达标；2 表示评估完成但未达标；1 表示执行失败。默认 mAP50 阈值为 0.8，可配置。每次运行创建独立目录，不覆盖历史报告。取消或失败可能留下部分诊断文件，不应作为完成报告使用。

默认 conf=0.001 是评估时保留预测的置信度下限；NMS IoU 参数不是计算 mAP50 时的匹配阈值。Precision/Recall 使用框架的汇总口径。mAP50 是匹配 IoU=0.5 下的平均精度，mAP50-95 汇总多个匹配阈值；这些指标不等于简单的“预测正确百分比”。

## 功能边界和文件组成

只支持当前 Ultralytics 能加载的标准目标检测 `.pt` 模型；自定义网络需要额外模块，不能保证兼容。暂不支持 ONNX、分类、分割、姿态、旋转框、训练或多模型对比。

`app.py` 为桌面界面，通过独立进程调用 `evaluate_model.py`；`dataset_checks.py` 检查目录和标注；`common.py` 提供公共文件操作；`reporting.py` 输出网页报告。原有单图、批量荔枝推理入口没有作为本工具功能入口。原始权重单独保存在 `examples/models/`，仅作可选示例；包内没有真实测试集。

## 第三方组件与权利说明

本工具由原交付包的评估程序改造，使用 Ultralytics、PyTorch（间接依赖）、OpenCV、PyYAML、Pillow 以及 Python/Tkinter。具体发行版本及其依赖以安装环境为准，安装成功后建议保存 `python -m pip freeze`。

原包记录的 Ultralytics 及示例权重许可为 AGPL-3.0；本工具未提供新的企业授权，也未核实原作者、数据或权重的权利归属。对本工具应用代码的最终许可、第三方 notices、AGPL 对应源码义务或企业授权，应由权利人结合实际部署方式整理，本文不构成许可合规证明。登记材料应如实区分自有代码和第三方组件。

官方参考：https://docs.ultralytics.com/modes/val/ 、https://docs.ultralytics.com/datasets/detect/ 、https://www.ultralytics.com/license 。
