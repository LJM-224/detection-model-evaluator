"""目标检测模型评估与分析软件的桌面入口。"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

ROOT = Path(__file__).resolve().parent
FROZEN = getattr(sys, 'frozen', False)


class EvaluatorApp:
    """管理模型评估任务及其可视化操作界面。"""

    def __init__(self, window):
        """创建参数表单、评估日志和结果操作按钮。"""
        self.window = window
        self.process = None
        self.report_dir = None
        self.events = queue.Queue()
        self.fields = {}
        window.title('目标检测模型评估与分析软件 V1.1')
        window.geometry('980x800')
        window.minsize(820, 680)
        panel = ttk.Frame(window, padding=22)
        panel.pack(fill='both', expand=True)
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text='目标检测模型评估', font=('Microsoft YaHei UI', 21, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))
        ttk.Label(panel, text='选择 YOLO .pt 模型和带标注测试集，生成指标、曲线和逐图错例报告。').grid(row=1, column=0, columnspan=3, sticky='w', pady=(0, 15))
        self.path_field(panel, 2, 'weights', '模型权重 .pt', '', 'model')
        self.path_field(panel, 3, 'input-dir', '数据集根目录', '', 'directory')
        self.path_field(panel, 4, 'data', 'data.yaml（可选）', '', 'yaml')
        output_root = Path.home() / 'DetectionModelEvaluator' if FROZEN else ROOT
        self.path_field(panel, 5, 'output-dir', '报告保存目录', str(output_root/'results'), 'directory')
        ttk.Label(panel, text='填写 data.yaml 后优先使用其路径与类别；否则自动识别目录并使用模型类别。').grid(row=6, column=0, columnspan=3, sticky='w', pady=6)
        params = ttk.LabelFrame(panel, text='评估参数', padding=10)
        params.grid(row=7, column=0, columnspan=3, sticky='ew', pady=10)
        specs = [('split','数据划分','test'), ('imgsz','图片尺寸','640'), ('batch','批大小','1'),
                 ('conf','评估置信度下限','0.001'), ('iou','NMS IoU','0.7'), ('require-map50','mAP50 达标阈值','0.8'),
                 ('diagnostic-conf','错例置信度（可选）',''), ('device','设备（cpu / 0）','cpu')]
        for index, (key, title, default) in enumerate(specs):
            row, col = divmod(index, 4)
            box = ttk.Frame(params, padding=4)
            box.grid(row=row, column=col, sticky='ew')
            params.columnconfigure(col, weight=1)
            ttk.Label(box, text=title).pack(anchor='w')
            value = tk.StringVar(value=default)
            self.fields[key] = value
            if key == 'split':
                ttk.Combobox(box, textvariable=value, values=['test','val'], state='readonly', width=15).pack(fill='x')
            else:
                ttk.Entry(box, textvariable=value, width=16).pack(fill='x')
        actions = ttk.Frame(panel)
        actions.grid(row=8, column=0, columnspan=3, sticky='ew', pady=8)
        self.start_button = ttk.Button(actions, text='开始评估', command=self.start)
        self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(actions, text='停止评估', command=self.cancel, state='disabled')
        self.cancel_button.pack(side='left', padx=8)
        self.report_button = ttk.Button(actions, text='查看报告与错例', command=self.open_report, state='disabled')
        self.report_button.pack(side='left', padx=8)
        self.folder_button = ttk.Button(actions, text='打开结果文件夹', command=self.open_folder, state='disabled')
        self.folder_button.pack(side='left')
        self.status = tk.StringVar(value='就绪。无目标图片也请保留同名空标签；仅有图片无法计算准确率指标。')
        ttk.Label(panel, textvariable=self.status, wraplength=880).grid(row=9, column=0, columnspan=3, sticky='w', pady=8)
        self.log = tk.Text(panel, height=16, wrap='word', state='disabled', font=('Consolas', 10))
        self.log.grid(row=10, column=0, columnspan=3, sticky='nsew')
        panel.rowconfigure(10, weight=1)
        window.protocol('WM_DELETE_WINDOW', self.close)
        window.after(150, self.poll)

    def path_field(self, panel, row, key, title, default, kind):
        """添加路径输入框和相应的文件或目录选择按钮。"""
        ttk.Label(panel, text=title).grid(row=row, column=0, sticky='w', padx=(0, 15), pady=6)
        self.fields[key] = tk.StringVar(value=default)
        ttk.Entry(panel, textvariable=self.fields[key]).grid(row=row, column=1, sticky='ew')
        from functools import partial
        ttk.Button(panel, text='选择…', command=partial(self.browse, key, kind)).grid(row=row, column=2, padx=(10, 0))

    def browse(self, key, kind):
        """选择本地模型、数据配置或数据目录并更新表单。"""
        if kind == 'directory':
            path = filedialog.askdirectory()
        else:
            extension = '*.pt' if kind == 'model' else '*.yaml *.yml'
            path = filedialog.askopenfilename(filetypes=[('支持的文件', extension)])
        if path:
            self.fields[key].set(path)

    def start(self):
        """检查表单并在独立进程中启动评估以保持窗口响应。"""
        if self.process is not None:
            return
        values = {key: value.get().strip() for key, value in self.fields.items()}
        if not values['weights'] or not values['output-dir'] or not (values['data'] or values['input-dir']):
            messagebox.showerror('信息未填写完整', '请选择模型、测试集或配置文件，以及结果保存目录。')
            return
        command = [sys.executable, '--evaluate'] if FROZEN else [sys.executable, '-u', str(ROOT/'evaluate_model.py')]
        for key, value in values.items():
            if value:
                command.extend(['--'+key, value])
        self.report_dir = None
        self.report_button.config(state='disabled')
        self.folder_button.config(state='disabled')
        self.log.config(state='normal')
        self.log.delete('1.0', 'end')
        self.log.config(state='disabled')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', cwd=None if FROZEN else ROOT, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except OSError as error:
            messagebox.showerror('无法启动', str(error))
            return
        self.start_button.config(state='disabled')
        self.cancel_button.config(state='normal')
        self.status.set('正在检查数据并评估模型，请等待；首次运行可能需要较长时间。')
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def read_output(self, process):
        """读取评估子进程输出并将消息发送给界面线程。"""
        for line in process.stdout:
            self.events.put(('line', line))
        self.events.put(('done', process.wait()))

    def poll(self):
        """在界面线程显示日志并更新评估完成状态。"""
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == 'line':
                self.log.config(state='normal')
                self.log.insert('end', value)
                self.log.see('end')
                self.log.config(state='disabled')
                if value.startswith('REPORT_DIR='):
                    self.report_dir = Path(value.strip().split('=', 1)[1])
            else:
                self.process = None
                self.start_button.config(state='normal')
                self.cancel_button.config(state='disabled')
                if value in (0, 2) and self.report_dir and (self.report_dir/'report.html').is_file():
                    payload = json.loads((self.report_dir/'metrics.json').read_text(encoding='utf-8'))
                    result = '达标' if payload['acceptance']['passed'] else '未达标'
                    self.status.set(f"评估完成：mAP50={payload['metrics']['map50']:.4f}，{result}。报告已自动保存。")
                    self.report_button.config(state='normal')
                    self.folder_button.config(state='normal')
                else:
                    self.status.set('评估停止或失败，请查看日志；若提示缺少依赖，请按使用说明安装。')
        self.window.after(150, self.poll)

    def cancel(self):
        """停止当前评估进程并保留已生成的诊断文件。"""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.status.set('正在停止评估…')

    def open_report(self):
        """在默认浏览器打开含指标和预测图的本地报告。"""
        if self.report_dir:
            webbrowser.open((self.report_dir/'report.html').as_uri())

    def open_folder(self):
        """打开本次评估的结果文件夹以访问导出的全部文件。"""
        if self.report_dir:
            if os.name == 'nt':
                os.startfile(self.report_dir)
            else:
                webbrowser.open(self.report_dir.as_uri())

    def close(self):
        """在关闭窗口前停止活动任务并释放界面。"""
        self.cancel()
        self.window.destroy()


def main():
    """启动目标检测模型评估桌面应用。"""
    window = tk.Tk()
    EvaluatorApp(window)
    window.mainloop()


if __name__ == '__main__':
    if FROZEN:
        from multiprocessing import freeze_support

        freeze_support()
        if sys.argv[1:2] == ['--evaluate']:
            del sys.argv[1]
            from evaluate_model import main as evaluate

            raise SystemExit(evaluate())
    main()
