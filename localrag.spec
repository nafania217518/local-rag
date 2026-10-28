# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_dynamic_libs

a = Analysis(
    ["src/localrag/app.py"],
    pathex=["src"],
    binaries=collect_dynamic_libs("sqlite_vec"),  # no community hook for sqlite-vec; vec0 lib must be collected by hand
    datas=[("src/localrag/static", "localrag/static")],
    hiddenimports=["webview"],  # app.py imports it via importlib
    excludes=["pymupdf.layout", "onnxruntime", "numpy", "networkx", "psutil"],  # optional layout stack, 170 MB, unused
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, name="LocalRAG", console=False)
