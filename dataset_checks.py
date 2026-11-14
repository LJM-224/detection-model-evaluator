"""检查目录型 YOLO 检测数据集，避免无效标注进入评估。"""
import math
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def inspect_dataset(images_dir: Path, labels_dir: Path, class_count: int) -> dict:
    """按相对子路径配对图片与标签并验证类别和归一化检测框。"""
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError('图片目录或标签目录不存在。')
    images = sorted(p for p in images_dir.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not images:
        raise ValueError('数据集中没有受支持的图片。')
    paired = set()
    boxes = 0
    empty = 0
    for image in images:
        relative = image.relative_to(images_dir).with_suffix('.txt')
        if relative in paired:
            raise ValueError(f'同一路径下存在共用标签的同名图片：{relative}')
        paired.add(relative)
        label = labels_dir / relative
        if not label.is_file():
            raise ValueError(f'缺少标注：{label}；无目标图片请提供空标签文件。')
        rows = label.read_text(encoding='utf-8-sig').splitlines()
        count = 0
        for line_number, row in enumerate(rows, 1):
            if not row.strip():
                continue
            try:
                values = [float(x) for x in row.split()]
                if len(values) != 5 or not all(math.isfinite(x) for x in values):
                    raise ValueError()
                category, x, y, width, height = values
                if category != int(category) or not 0 <= category < class_count:
                    raise ValueError()
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError()
                if x-width/2 < -1e-6 or y-height/2 < -1e-6 or x+width/2 > 1+1e-6 or y+height/2 > 1+1e-6:
                    raise ValueError()
            except ValueError:
                raise ValueError(f'标注格式或检测框无效：{label} 第 {line_number} 行') from None
            count += 1
        boxes += count
        empty += count == 0
    extras = sorted(str(p.relative_to(labels_dir)) for p in labels_dir.rglob('*.txt') if p.relative_to(labels_dir) not in paired)
    if boxes == 0:
        raise ValueError('整个数据集没有真实目标，无法可靠评估 mAP。')
    return dict(image_count=len(images), label_count=len(paired), object_count=boxes,
                empty_image_count=empty, missing_label_count=0,
                extra_label_count=len(extras), extra_label_examples=extras[:20])


def find_split(root: Path, split: str) -> tuple[Path, Path]:
    """识别常见的三种图片与标签目录布局。"""
    candidates = [(root/split/'images', root/split/'labels'),
                  (root/'images'/split, root/'labels'/split),
                  (root/'images', root/'labels')]
    for images, labels in candidates:
        if images.is_dir() and labels.is_dir():
            return images.resolve(), labels.resolve()
    raise ValueError('未找到数据集，请使用 images/test 与 labels/test、test/images 与 test/labels，或 images 与 labels。')


def labels_for_images(images: Path) -> Path:
    """将标准图片目录转换为对应的标签目录。"""
    parts = list(images.parts)
    indexes = [i for i, part in enumerate(parts) if part == 'images']
    if not indexes:
        raise ValueError('data.yaml 的图片路径必须包含 images 目录以定位 labels。')
    parts[indexes[-1]] = 'labels'
    return Path(*parts)
