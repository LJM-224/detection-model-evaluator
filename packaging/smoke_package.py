"""通过打包后的程序及临时测试数据验证完整评估流程。"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def main():
    """生成随机权重和合成图片，再验证打包程序能输出评估与诊断报告。"""
    executable = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weights = root / "test.pt"
        YOLO("yolo26n.yaml").save(weights)
        images = root / "images/test"
        labels = root / "labels/test"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (96, 72), "white").save(images / "sample.png")
        (labels / "sample.txt").write_text("0 .5 .5 .25 .25\n", encoding="utf-8")
        result = subprocess.run(
            [str(executable), "--evaluate", "--weights", str(weights), "--input-dir", str(root),
             "--output-dir", str(root / "results"), "--device", "cpu", "--imgsz", "160"],
            cwd=root,
            env={**os.environ, "PYTHONUTF8": "1"},
            timeout=180,
        )
        if result.returncode not in (0, 2):
            raise RuntimeError(f"打包程序评估失败，退出码：{result.returncode}")
        output = next((root / "results").iterdir())
        predictions = json.loads((output / "image_predictions.json").read_text(encoding="utf-8"))
        if set(predictions["images"]) != {"sample.png"} or not (output / "diagnostics.html").is_file():
            raise RuntimeError("打包程序没有生成完整的逐图评估报告。")


if __name__ == "__main__":
    main()
