"""从同一次模型验证收集逐图检测框，并导出可筛选的错例报告。"""

import csv
import html
from collections import Counter
from pathlib import Path

from common import write_json


ERROR_NAMES = {"missed": "漏检", "false_positive": "误检", "confusion": "类别混淆"}
MATCH_IOU = 0.50


def capture_validator(image_root, records):
    """创建保留验证指标并记录每张图片检测框的验证器。"""
    from ultralytics.models.yolo.detect import DetectionValidator

    class DiagnosticValidator(DetectionValidator):
        """为标准检测验证器增加逐图结果采集。"""

        def update_metrics(self, preds, batch):
            """更新标准指标并保存原图坐标下的标注和预测。"""
            super().update_metrics(preds, batch)
            for index, pred in enumerate(preds):
                prepared = self._prepare_batch(index, batch)
                actual = self.scale_preds(prepared, prepared)
                predicted = self.scale_preds(pred, prepared)
                path = Path(prepared["im_file"]).relative_to(image_root).as_posix()
                height, width = prepared["ori_shape"]
                records[path] = {
                    "width": width,
                    "height": height,
                    "ground_truth": [
                        {"class_id": int(category), "xyxy": box}
                        for category, box in zip(actual["cls"].tolist(), actual["bboxes"].tolist())
                    ],
                    "predictions": [
                        {"class_id": int(category), "xyxy": box, "confidence": score}
                        for category, box, score in zip(
                            predicted["cls"].tolist(), predicted["bboxes"].tolist(), predicted["conf"].tolist()
                        )
                    ],
                }

    return DiagnosticValidator


def box_iou(first, second):
    """计算两个原图坐标检测框的交并比。"""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def find_errors(actual, predicted, confidence):
    """先匹配同类框再匹配异类框，并按一对一规则识别剩余错例。"""
    active = [index for index, item in enumerate(predicted) if item["confidence"] >= confidence]
    candidates = []
    for truth_index, truth in enumerate(actual):
        for pred_index in active:
            prediction = predicted[pred_index]
            overlap = box_iou(truth["xyxy"], prediction["xyxy"])
            if overlap >= MATCH_IOU:
                different_class = truth["class_id"] != prediction["class_id"]
                candidates.append((different_class, -overlap, -prediction["confidence"], truth_index, pred_index))

    unmatched_truth = set(range(len(actual)))
    unmatched_pred = set(active)
    errors = []
    for different_class, negative_iou, _, truth_index, pred_index in sorted(candidates):
        if truth_index not in unmatched_truth or pred_index not in unmatched_pred:
            continue
        unmatched_truth.remove(truth_index)
        unmatched_pred.remove(pred_index)
        if different_class:
            errors.append({"type": "confusion", "truth_index": truth_index, "prediction_index": pred_index,
                           "iou": -negative_iou})
    errors.extend({"type": "missed", "truth_index": index} for index in sorted(unmatched_truth))
    errors.extend({"type": "false_positive", "prediction_index": index} for index in sorted(unmatched_pred))
    return errors


PAGE_STYLE = """
body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#17324d;background:#f5f7fb}
a{color:#12589b}table{border-collapse:collapse;width:100%;background:white;overflow-wrap:anywhere}
td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}label{display:inline-block;margin:0 20px 14px 0}
img{display:block;max-width:100%;height:auto}.truth{color:#147347}.prediction{color:#aa380f}
"""


def write_page(path, title, body):
    """将内容写入无需联网即可浏览的诊断网页。"""
    path.write_text(
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>{PAGE_STYLE}</style>'
        f'<main><h1>{html.escape(title)}</h1>{body}</main></html>',
        encoding="utf-8",
    )


def write_case_preview(source, record, confidence, destination):
    """生成叠加真实框和诊断阈值以上预测框的便携预览图。"""
    from PIL import Image, ImageDraw
    from ultralytics.utils.patches import imread

    pixels = imread(str(source))
    if pixels is None:
        raise ValueError(f"无法读取错例图片：{source}")
    image = Image.fromarray(pixels[..., ::-1])
    if image.size != (record["width"], record["height"]):
        raise ValueError(f"错例图片尺寸与评估时不一致：{source}")
    image.thumbnail((1400, 1400))
    sx, sy = image.width / record["width"], image.height / record["height"]
    draw = ImageDraw.Draw(image)
    for key, prefix, color, stroke in (
        ("ground_truth", "G", "#147347", 4), ("predictions", "P", "#c74719", 2)
    ):
        for index, item in enumerate(record[key]):
            if item.get("confidence", 1.0) < confidence:
                continue
            x1, y1, x2, y2 = item["xyxy"]
            draw.rectangle((x1*sx, y1*sy, x2*sx, y2*sy), outline=color, width=stroke)
            text = f"{prefix}{index + 1}"
            width = draw.textlength(text)
            x = max(0, min(x1*sx, image.width - width - 6))
            y = max(0, min(y1*sy + (14 if prefix == "P" else -14), image.height - 14))
            draw.rectangle((x, y, x + width + 6, y + 14), fill=color)
            draw.text((x + 3, y), text, fill="white")
    image.save(destination, quality=88)


def error_rows(relative, record, names):
    """生成含框编号、类别、置信度和匹配交并比的错例明细。"""
    for error in record["errors"]:
        truth_index = error.get("truth_index")
        prediction_index = error.get("prediction_index")
        truth = record["ground_truth"][truth_index] if truth_index is not None else None
        prediction = record["predictions"][prediction_index] if prediction_index is not None else None
        yield {
            "image": relative,
            "error": ERROR_NAMES[error["type"]],
            "truth_box": f"G{truth_index + 1}" if truth is not None else "",
            "truth_class": names[truth["class_id"]] if truth is not None else "",
            "prediction_box": f"P{prediction_index + 1}" if prediction is not None else "",
            "prediction_class": names[prediction["class_id"]] if prediction is not None else "",
            "confidence": f'{prediction["confidence"]:.4f}' if prediction is not None else "",
            "iou": f'{error["iou"]:.4f}' if "iou" in error else "",
        }


def spreadsheet_text(value):
    """将可能被表格软件解释为公式的文本标记为普通文本。"""
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


FILTER_SCRIPT = """
const filters = document.querySelectorAll('select');
const rows = document.querySelectorAll('#cases tbody tr');
function filterCases() {
    // 根据错误类型和类别的组合显示对应图片。
    const [type, category] = Array.from(filters, filter => filter.value);
    let total = 0;
    for (const row of rows) {
        const keys = row.dataset.keys.split(' ');
        row.hidden = !(keys.includes(type + ':' + category));
        total += Number(!row.hidden);
    }
    document.querySelector('#visible').textContent = total + ' 张错例图片';
}
for (const filter of filters) filter.addEventListener('change', filterCases);
filterCases();
"""


def export_diagnostics(records, image_root, names, confidence, output_dir):
    """导出全量预测、错例清单、筛选索引和逐图预览。"""
    previews = output_dir / "diagnostics"
    previews.mkdir()
    counts = Counter()
    cases = []
    escape = html.escape
    columns = ("image", "error", "truth_box", "truth_class", "prediction_box", "prediction_class", "confidence", "iou")
    with (output_dir / "diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, (relative, record) in enumerate(sorted(records.items()), 1):
            errors = find_errors(record["ground_truth"], record["predictions"], confidence)
            record["errors"] = errors
            if not errors:
                continue
            keys = {"all:all"}
            totals = Counter(error["type"] for error in errors)
            counts.update(totals)
            for error in errors:
                error_type = error["type"]
                keys.add(f"{error_type}:all")
                for field, boxes in (("truth_index", "ground_truth"), ("prediction_index", "predictions")):
                    if field in error:
                        category = record[boxes][error[field]]["class_id"]
                        keys.update((f"{error_type}:{category}", f"all:{category}"))
            image_path = f"{index}.jpg"
            page_path = f"{index}.html"
            write_case_preview(image_root / relative, record, confidence, previews / image_path)
            details = list(error_rows(relative, record, names))
            writer.writerows({key: spreadsheet_text(value) for key, value in row.items()} for row in details)
            table_rows = "".join(
                "<tr>" + "".join(f"<td>{escape(row[key])}</td>" for key in columns[1:]) + "</tr>"
                for row in details
            )
            write_page(
                previews / page_path, relative,
                '<p><a href="../diagnostics.html">返回错例列表</a></p>'
                '<p><strong class="truth">绿色 G · 真实框</strong>　'
                '<strong class="prediction">橙色 P · 预测框</strong></p>'
                f'<img src="{image_path}" alt="真实框和预测框叠加图"><h2>错例明细</h2>'
                '<table><thead><tr><th>错误类型</th><th>真实框</th><th>真实类别</th>'
                '<th>预测框</th><th>预测类别</th><th>置信度</th><th>IoU</th></tr></thead>'
                f'<tbody>{table_rows}</tbody></table>',
            )
            cases.append(
                f'<tr data-keys="{" ".join(sorted(keys))}">'
                f'<td><a href="diagnostics/{page_path}">{escape(relative)}</a></td>'
                + "".join(f'<td>{totals[kind]}</td>' for kind in ERROR_NAMES) + '</tr>'
            )
    summary = {"confidence": confidence, "matching_iou": MATCH_IOU, "image_count": len(records),
               "error_image_count": len(cases), "counts": {kind: counts[kind] for kind in ERROR_NAMES}}
    write_json({"schema_version": 1, "coordinate_format": "xyxy_pixels", "summary": summary,
                "class_names": names, "images": records}, output_dir / "image_predictions.json")
    error_options = "".join(f'<option value="{kind}">{name}</option>' for kind, name in ERROR_NAMES.items())
    class_options = "".join(f'<option value="{key}">{escape(name)}</option>' for key, name in names.items())
    write_page(
        output_dir / "diagnostics.html", "逐图错例分析",
        '<p><a href="report.html">返回评估报告</a>　<a href="diagnostics.csv" download>下载错例 CSV</a></p>'
        f'<p>诊断置信度 ≥ {confidence:g}，匹配 IoU ≥ {MATCH_IOU:g}。'
        '优先匹配同类框，再匹配剩余异类框；每个框最多匹配一次。</p>'
        f'<label>错误类型 <select><option value="all">全部</option>{error_options}</select></label>'
        f'<label>类别 <select><option value="all">全部</option>{class_options}</select></label>'
        f'<p id="visible" role="status">{len(cases)} 张错例图片</p>'
        '<table id="cases"><thead><tr><th>图片</th><th>漏检</th><th>误检</th><th>类别混淆</th></tr></thead>'
        f'<tbody>{"".join(cases)}</tbody></table><script>{FILTER_SCRIPT}</script>',
    )
    return summary
