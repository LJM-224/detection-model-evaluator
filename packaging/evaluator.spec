# 为 Windows 和 Linux 生成包含运行依赖的免安装应用目录。
import sys
from importlib.metadata import distributions
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

sys.setrecursionlimit(sys.getrecursionlimit() * 5)
root = Path(SPECPATH).parent
datas, binaries, hiddenimports = collect_all("ultralytics", include_py_files=False)
datas.append((str(root / "README.md"), "."))
for distribution in distributions():
    name = distribution.metadata["Name"]
    for entry in distribution.files or ():
        if ".dist-info/" not in entry.as_posix():
            continue
        relative = Path(*entry.parts[1:])
        if "licenses" in relative.parts or relative.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
            source = distribution.locate_file(entry)
            if source.is_file():
                datas.append((str(source), str(Path("licenses", name, relative.parent))))

analysis = Analysis(
    [str(root / "app.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    module_collection_mode={"ultralytics": "pyz+py"},
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True,
          name="DetectionModelEvaluator", console=True, upx=False)
bundle = COLLECT(exe, analysis.binaries, analysis.datas, upx=False, name="DetectionModelEvaluator")
