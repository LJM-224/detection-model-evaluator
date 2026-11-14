"""验证数据检查、类别指标映射和报告导出等关键行为。"""
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dataset_checks import inspect_dataset, find_split
from evaluate_model import build_metrics_payload, prepare_dataset, write_metrics_files
from reporting import render_report


class EvaluatorTests(unittest.TestCase):
    """覆盖通用评估中容易产生错误结果的边界条件。"""

    def setUp(self):
        """建立独立临时目录避免测试改变用户数据。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root/'images/test'
        self.labels = self.root/'labels/test'
        self.images.mkdir(parents=True)
        self.labels.mkdir(parents=True)

    def sample(self, relative='a', label='0 0.5 0.5 0.2 0.2'):
        """生成真实可解码的图片及其对应标签文件。"""
        from PIL import Image
        image = self.images/(relative+'.png')
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32,32), 'white').save(image)
        annotation = self.labels/(relative+'.txt')
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(label, encoding='utf-8')

    def test_relative_pairing(self):
        """确认不同子目录的同名图片不会错误共用标签。"""
        self.sample('a/item')
        self.sample('b/item')
        (self.labels/'b/item.txt').unlink()
        with self.assertRaisesRegex(ValueError, '缺少标注'):
            inspect_dataset(self.images, self.labels, 1)

    def test_invalid_labels(self):
        """确认非法类别、非有限数值和越界检测框被拒绝。"""
        for text in ['2 0.5 0.5 0.2 0.2', '0 nan 0.5 0.2 0.2', '0 0.1 0.5 0.9 0.2', '0 0.5 0.5 0 0.2']:
            with self.subTest(text=text):
                self.sample(label=text)
                with self.assertRaises(ValueError):
                    inspect_dataset(self.images, self.labels, 1)

    def test_negative_and_positive(self):
        """确认空标签可作为负样本并正确统计目标总数。"""
        self.sample('positive')
        self.sample('negative', '')
        stats = inspect_dataset(*find_split(self.root, 'test'), 1)
        self.assertEqual((stats['image_count'],stats['object_count'],stats['empty_image_count']), (2,1,1))

    def test_yaml_classes_and_paths(self):
        """确认配置相对路径正确解析且类别语义不匹配会被拒绝。"""
        self.sample()
        config = self.root/'data.yaml'
        config.write_text('path: .\ntest: images/test\nnames: [car]\n', encoding='utf-8')
        _, _, stats = prepare_dataset(self.root, config, 'test', self.root, {0:'car'})
        self.assertEqual(stats['image_count'], 1)
        with self.assertRaisesRegex(ValueError, '类别名称'):
            prepare_dataset(self.root, config, 'test', self.root, {0:'person'})

    def test_sparse_class_metrics_and_report(self):
        """确认仅部分类别参与评估时指标不会错位且报告正确转义文本。"""
        import numpy as np
        weights = self.root/'model.pt'
        weights.write_bytes(b'test')
        box = SimpleNamespace(mp=.8, mr=.7, map50=.9, map=.6, p=[.8], r=[.7],
                              ap50=[.9], all_ap=np.array([[.6,.6]]), ap_class_index=[2])
        metrics = SimpleNamespace(box=box, speed={'inference':2}, save_dir=self.root/'test_result')
        model = SimpleNamespace(names={0:'absent',1:'other',2:'<car>'})
        args = argparse.Namespace(split='test',imgsz=640,conf=.001,iou=.7,batch=1,device='cpu',require_map50=.95)
        payload = build_metrics_payload(metrics, model, weights, None, self.root/'data.yaml',
                                        {'names':model.names}, {'image_count':1,'label_count':1}, args)
        classes = payload['metrics']['per_class']
        self.assertEqual([x['class_id'] for x in classes], [0,1,2])
        self.assertIsNone(classes[0]['map50'])
        self.assertEqual(classes[2]['map50'], .9)
        self.assertFalse(payload['acceptance']['passed'])
        write_metrics_files(payload, self.root)
        render_report(payload, self.root)
        self.assertIn('&lt;car&gt;', (self.root/'report.html').read_text(encoding='utf-8'))
        self.assertTrue((self.root/'metrics.csv').is_file())


if __name__ == '__main__':
    unittest.main()
