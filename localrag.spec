# -*- mode: python -*-
import sys
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
exe = EXE(pyz, a.scripts, exclude_binaries=True, name="LocalRAG", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="LocalRAG")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="LocalRAG.app", bundle_identifier="dev.localrag",
                 info_plist={"NSPrincipalClass": "NSApplication", "NSHighResolutionCapable": True})
