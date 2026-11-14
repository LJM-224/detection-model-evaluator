#!/usr/bin/env python3
"""使用带真实标注的通用目标检测验证集评估检测模型并输出验收指标。"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from common import calculate_sha256, write_json
from dataset_checks import EXTENSIONS, inspect_dataset, find_split, labels_for_images
from diagnostics import capture_validator, export_diagnostics
from reporting import render_report


def parse_args() -> argparse.Namespace:
    """解析目标检测验证集评估所需的命令行参数。"""
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="通用目标检测 YOLO 验证集评估")
    parser.add_argument(
        "--weights",
        required=True,
        help="YOLO 权重路径",
    )
    parser.add_argument(
        "--input-dir",
        default=str(package_dir / "input_images"),
        help="带标签数据集根目录，支持 images/test 或 test/images 等布局",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="可选的 data.yaml 路径；不填写时自动根据 input-dir 生成",
    )
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument(
        "--output-dir",
        default=str(package_dir / "output" / "evaluation"),
        help="评估结果输出目录",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="评估输入尺寸")
    parser.add_argument("--conf", type=float, default=0.001, help="评估置信度下限")
    parser.add_argument("--diagnostic-conf", type=float, default=None,
                        help="逐图错例置信度阈值，默认取 0.25 和评估置信度下限中的较大值")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU 阈值")
    parser.add_argument("--batch", type=int, default=1, help="评估批大小")
    parser.add_argument("--device", default=None, help="评估设备，例如 0、cpu 或 cuda:0")
    parser.add_argument(
        "--require-map50",
        type=float,
        default=0.80,
        help="mAP50 验收阈值",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    """校验评估路径和数值参数并返回规范化路径。"""
    weights_path = Path(args.weights).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve() if args.data else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if weights_path.suffix.lower() != ".pt":
        raise ValueError("第一版仅支持 Ultralytics YOLO 的 .pt 权重。")
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重不存在：{weights_path}")
    if data_path is not None and not data_path.is_file():
        raise FileNotFoundError(f"数据集配置不存在：{data_path}")
    if data_path is None and not input_dir.is_dir():
        raise FileNotFoundError(f"验证集目录不存在：{input_dir}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf 必须位于 0 到 1 之间")
    diagnostic_conf = getattr(args, "diagnostic_conf", None)
    if diagnostic_conf is not None and not args.conf <= diagnostic_conf <= 1.0:
        raise ValueError("错例置信度必须介于评估置信度下限和 1 之间")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou 必须位于 0 到 1 之间")
    if not 0.0 <= args.require_map50 <= 1.0:
        raise ValueError("require-map50 必须位于 0 到 1 之间")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("imgsz 和 batch 必须大于 0")
    return weights_path, input_dir, data_path, output_dir


def prepare_dataset(input_dir: Path, data_path: Path | None, split: str,
                    output_dir: Path, class_names: dict) -> tuple:
    """检查目录或配置文件中的数据并生成仅用于本次评估的绝对路径配置。"""
    import yaml
    from PIL import Image
    names = {int(k): str(v) for k, v in class_names.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError("模型类别编号必须从零连续排列。")
    if data_path:
        config = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("data.yaml 必须是映射。")
        configured_names = config.get("names")
        if isinstance(configured_names, list):
            configured_names = dict(enumerate(configured_names))
        if not isinstance(configured_names, dict) or {int(k):str(v) for k,v in configured_names.items()} != names:
            raise ValueError("数据集类别名称与编号必须和模型完全一致。")
        root = Path(str(config.get("path") or ".")).expanduser()
        if not root.is_absolute():
            root = data_path.parent / root
        selected = config.get(split)
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError(f"data.yaml 缺少 {split} 图片目录；第一版仅支持目录型划分。")
        images = Path(selected).expanduser()
        if not images.is_absolute():
            images = root / images
        images = images.resolve()
        labels = labels_for_images(images)
    else:
        images, labels = find_split(input_dir, split)
    stats = inspect_dataset(images, labels, len(names))
    for image in images.rglob("*"):
        if image.suffix.lower() in EXTENSIONS and image.is_file():
            with Image.open(image) as opened:
                opened.verify()
    config = {"path": str(images.parent), "train": str(images),
              "val": str(images), "test": str(images), "names": names}
    resolved = output_dir / "resolved_data.yaml"
    resolved.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return resolved, config, stats


def to_float_list(values: Any) -> list[float]:
    """将NumPy或张量形式的指标序列转换为普通浮点数列表。"""
    if values is None:
        return []
    return [float(value) for value in values]


def build_metrics_payload(
    metrics: Any,
    model: Any,
    weights_path: Path,
    data_path: Path | None,
    resolved_data_path: Path,
    config: dict[str, Any],
    dataset_check: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """将Ultralytics验证指标整理为稳定的目标检测验收报告。"""
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    class_precision = to_float_list(getattr(metrics.box, "p", None))
    class_recall = to_float_list(getattr(metrics.box, "r", None))
    class_map50 = to_float_list(getattr(metrics.box, "ap50", None))
    class_map50_95 = to_float_list(metrics.box.all_ap.mean(axis=1))
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in model.names.items():
        evaluated_ids = [int(v) for v in metrics.box.ap_class_index]
        index = evaluated_ids.index(int(class_id)) if int(class_id) in evaluated_ids else -1
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name),
                "precision": class_precision[index] if 0 <= index < len(class_precision) else None,
                "recall": class_recall[index] if 0 <= index < len(class_recall) else None,
                "map50": class_map50[index] if 0 <= index < len(class_map50) else None,
                "map50_95": class_map50_95[index] if 0 <= index < len(class_map50_95) else None,
            }
        )
    return {
        "schema_version": 1,
        "success": True,
        "task": "object_detection_evaluation",
        "model": {
            "weights_file": weights_path.name,
            "weights_sha256": calculate_sha256(weights_path),
        },
        "dataset": {
            "source_data_yaml": str(data_path) if data_path else None,
            "resolved_data_yaml": str(resolved_data_path),
            "split": args.split,
            "image_count": dataset_check.get("image_count"),
            "label_count": dataset_check.get("label_count"),
            "object_count": dataset_check.get("object_count"),
            "empty_image_count": dataset_check.get("empty_image_count"),
            "missing_label_count": dataset_check.get("missing_label_count", 0),
            "extra_label_count": dataset_check.get("extra_label_count", 0),
            "extra_label_examples": dataset_check.get("extra_label_examples", []),
            "class_names": config.get("names"),
        },
        "parameters": {
            "image_size": args.imgsz,
            "confidence_floor": args.conf,
            "nms_iou_threshold": args.iou,
            "matching_iou_threshold_for_map50": 0.50,
            "batch": args.batch,
            "device": args.device or "auto",
        },
        "iou_definition": {
            "formula": "IoU = area(prediction ∩ ground truth) / area(prediction ∪ ground truth)",
            "description": "类别正确且预测框与真实框的IoU不低于0.5时参与mAP50的正确匹配判定。",
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "per_class": per_class,
        },
        "acceptance": {
            "required_map50": args.require_map50,
            "passed": map50 >= args.require_map50,
            "difference": map50 - args.require_map50,
        },
        "speed_ms_per_image": {
            key: round(float(value), 3) for key, value in metrics.speed.items()
        },
        "plot_directory": str(Path(metrics.save_dir).resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_metrics_files(payload: dict[str, Any], output_dir: Path) -> None:
    """将评估结果写成JSON、CSV、文本和独立mAP50文件。"""
    write_json(payload, output_dir / "metrics.json")
    metrics = payload["metrics"]
    acceptance = payload["acceptance"]
    status = "达标" if acceptance["passed"] else "未达标"
    text_lines = [
        "通用目标检测目标检测模型评估结果",
        f"数据划分：{payload['dataset']['split']}",
        f"图片数量：{payload['dataset']['image_count']}",
        f"标签文件数量：{payload['dataset']['label_count']}",
        f"缺少标签：{payload['dataset']['missing_label_count']}",
        f"多余标签：{payload['dataset']['extra_label_count']}",
        f"Precision：{metrics['precision']:.6f}",
        f"Recall：{metrics['recall']:.6f}",
        f"mAP50：{metrics['map50']:.6f}",
        f"mAP50-95：{metrics['map50_95']:.6f}",
        f"验收要求：mAP50 ≥ {acceptance['required_map50']:.2%}",
        f"验收结论：{status}",
        "IoU公式：IoU = 预测框与真实框交集面积 / 预测框与真实框并集面积",
    ]
    (output_dir / "metrics.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    map_text = (
        f"mAP50={metrics['map50']:.6f}\n"
        f"required_mAP50={acceptance['required_map50']:.6f}\n"
        f"passed={'YES' if acceptance['passed'] else 'NO'}\n"
    )
    (output_dir / "map.txt").write_text(map_text, encoding="utf-8")
    fieldnames = [
        "scope",
        "class_name",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "required_map50",
        "passed",
    ]
    rows = [
        {
            "scope": "overall",
            "class_name": "all",
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "required_map50": acceptance["required_map50"],
            "passed": acceptance["passed"],
        }
    ]
    for class_metrics in metrics["per_class"]:
        rows.append(
            {
                "scope": "class",
                "class_name": class_metrics["class_name"],
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "map50": class_metrics["map50"],
                "map50_95": class_metrics["map50_95"],
                "required_map50": acceptance["required_map50"],
                "passed": (
                    class_metrics["map50"] >= acceptance["required_map50"]
                    if class_metrics["map50"] is not None
                    else False
                ),
            }
        )
    with (output_dir / "metrics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_prediction_images_to_png(result_dir: Path) -> list[str]:
    """将验证过程生成的预测结果图转换为规范要求的PNG格式。"""
    from PIL import Image
    png_names: list[str] = []
    for source_path in sorted(result_dir.glob("*_pred.jpg")):
        target_path = source_path.with_suffix(".png")
        with Image.open(source_path) as image:
            image.save(target_path, format="PNG")
        png_names.append(target_path.name)
    return png_names


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """运行完整验证集评估并生成验收报告和可视化结果。"""
    weights_path, input_dir, data_path, output_root = validate_args(args)
    import tempfile
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("evaluation_%Y%m%d_%H%M%S_"), dir=output_root))
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    if model.task != "detect":
        raise ValueError("仅支持目标检测模型，不支持分类、分割、姿态或旋转框模型。")
    resolved_data_path, config, dataset_check = prepare_dataset(
        input_dir, data_path, args.split, output_dir, model.names)
    validation_args: dict[str, Any] = {
        "data": str(resolved_data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "batch": args.batch,
        "plots": True,
        "project": str(output_dir),
        "name": "test_result",
        "exist_ok": True,
        "verbose": False,
        "workers": 0,
    }
    if args.device:
        validation_args["device"] = args.device
    image_root = Path(config[args.split])
    records = {}
    metrics = model.val(validator=capture_validator(image_root, records), **validation_args)
    expected_images = {
        path.relative_to(image_root).as_posix()
        for path in image_root.rglob("*") if path.suffix.lower() in EXTENSIONS and path.is_file()
    }
    missing_images = expected_images - records.keys()
    if missing_images or len(records) != dataset_check["image_count"]:
        examples = "、".join(sorted(missing_images)[:5])
        raise ValueError(f"验证器未覆盖已检查的全部图片，缺少 {len(missing_images)} 张。{examples}")
    prediction_png_files = convert_prediction_images_to_png(Path(metrics.save_dir))
    payload = build_metrics_payload(
        metrics,
        model,
        weights_path,
        data_path,
        resolved_data_path,
        config,
        dataset_check,
        args,
    )
    payload["prediction_png_files"] = prediction_png_files
    payload["output_directory"] = str(output_dir)
    confidence = getattr(args, "diagnostic_conf", None)
    if confidence is None:
        confidence = max(0.25, args.conf)
    print(f"正在生成 {len(records)} 张图片的逐图诊断…")
    payload["diagnostics"] = export_diagnostics(records, image_root, config["names"], confidence, output_dir)
    write_metrics_files(payload, output_dir)
    render_report(payload, output_dir)
    return payload


def main() -> int:
    """执行目标检测模型评估并在终端显示mAP50验收结论。"""
    args = parse_args()
    try:
        payload = run_evaluation(args)
    except Exception as error:
        print(f"模型评估失败：{error}", file=sys.stderr)
        return 1
    metrics = payload["metrics"]
    acceptance = payload["acceptance"]
    print(f"评估完成：mAP50={metrics['map50']:.4f}")
    print(
        f"验收要求：mAP50≥{acceptance['required_map50']:.2%}，"
        f"结果={'达标' if acceptance['passed'] else '未达标'}"
    )
    print(f"评估目录：{payload['output_directory']}")
    print(f"REPORT_DIR={payload['output_directory']}")
    return 0 if acceptance["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
