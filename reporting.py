"""生成可离线浏览和打印的评估报告。"""
import html
from pathlib import Path


def render_report(payload: dict, output_dir: Path) -> None:
    """将总体指标、类别指标和本地结果图写入独立网页报告。"""
    escape = html.escape
    metrics = payload['metrics']
    cards = ''.join(f'<div><strong>{escape(name)}</strong><p>{metrics[key]:.4f}</p></div>' for name, key in [('Precision','precision'), ('Recall','recall'), ('mAP50','map50'), ('mAP50-95','map50_95')])
    rows = ''
    for item in metrics['per_class']:
        cells = ''.join('<td>未参与评估</td>' if item[key] is None else f'<td>{item[key]:.4f}</td>' for key in ['precision','recall','map50','map50_95'])
        rows += f'<tr><td>{item["class_id"]}</td><td>{escape(item["class_name"])}</td>{cells}</tr>'
    figures = ''
    for path in sorted((output_dir/'test_result').glob('*')):
        if path.suffix.lower() in {'.png', '.jpg'}:
            relative = path.relative_to(output_dir).as_posix()
            figures += f'<figure><figcaption>{escape(path.name)}</figcaption><a href="{escape(relative, quote=True)}"><img src="{escape(relative, quote=True)}" loading="lazy"></a></figure>'
    status = '达标' if payload['acceptance']['passed'] else '未达标'
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>目标检测模型评估报告</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#17324d;background:#f5f7fb}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.cards div{{background:white;padding:20px;min-width:150px;border-radius:12px}}p{{overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%;max-height:650px}}figure{{background:white;padding:16px;margin:20px 0}}@media print{{body{{background:white}}}}</style>
<h1>目标检测模型评估报告</h1><p>目标检测模型评估与分析软件 V1.0</p>
<p>模型：{escape(payload['model']['weights_file'])} · 图片：{payload['dataset']['image_count']} · 划分：{escape(payload['dataset']['split'])}</p>
<p>生成时间：{escape(payload['generated_at'])}</p><div class="cards">{cards}</div>
<p>mAP50 验收阈值：{payload['acceptance']['required_map50']:.2%}，本次结论：{status}。</p>
<p>mAP50 与 mAP50-95 使用不同匹配阈值；界面中的 IoU 参数是 NMS 阈值。Precision/Recall 采用框架汇总口径，并非固定置信度下的逐图统计。</p>
<h2>类别指标</h2><table><tr><th>ID</th><th>类别</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th></tr>{rows}</table>
<h2>可追溯信息</h2><p>模型 SHA256：{escape(payload['model']['weights_sha256'])}</p>
<p>输入尺寸：{payload['parameters']['image_size']}；置信度下限：{payload['parameters']['confidence_floor']}；NMS IoU：{payload['parameters']['nms_iou_threshold']}；设备：{escape(payload['parameters']['device'])}</p>
<p>详细配置与统计见同目录 metrics.json、metrics.csv 和 resolved_data.yaml。预测图为框架生成的批次样例，不保证覆盖所有图片。</p><h2>曲线与检测样例</h2>{figures}</html>'''
    (output_dir/'report.html').write_text(content, encoding='utf-8')
