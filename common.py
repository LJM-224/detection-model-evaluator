"""目标检测评估工具的通用文件操作。"""
import hashlib
import json
from pathlib import Path


def calculate_sha256(file_path: Path) -> str:
    """分块计算文件的摘要以记录评估所用模型。"""
    digest = hashlib.sha256()
    with file_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: dict, output_path: Path) -> None:
    """将结构化数据保存为可阅读的中文文本文件。"""
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
