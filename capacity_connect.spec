# capacity_connect.spec
# Build with: pyinstaller -y capacity_connect.spec

import os, sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE

block_cipher = None

# --- libs that need data/hidden imports gathered ---
hiddenimports, datas, binaries = [], [], []
for pkg in ["pandas", "plotly", "dash"]:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# >>> ADD THIS: bundle dash_svg data (package-info.json etc.)
try:
    d, b, h = collect_all("dash_svg")   # import name is dash_svg
    datas += d; binaries += b; hiddenimports += h
except Exception:
    pass
# <<< ADD END

# --- your internal packages/modules ---
hiddenimports += collect_submodules("pages")
hiddenimports += collect_submodules("callbacks_pkg")
hiddenimports += collect_submodules("plan_detail")
hiddenimports += [
    "auth","ba_rollup_plan","cap_db","cap_store","capacity_core",
    "common","plan_store","planning_workspace","router","app_instance","packaging_paths"
]

# --- data files (only (src, dest) tuples) ---
datas += [
    (".flask_secret_key", "."),
    ("capability.sqlite3", "."),
    ("assets", "assets"),
]

# --- optional: conda DLL for pyexpat (if you hit DLL errors) ---
def maybe_add_dll(name: str):
    roots = filter(None, [os.environ.get("CONDA_PREFIX"), sys.base_prefix, sys.prefix])
    for r in roots:
        p = Path(r)
        for c in [p/"Library"/"bin"/name, p/"DLLs"/name, p/name]:
            if c.exists():
                binaries.append((str(c), "."))
                return
maybe_add_dll("libexpat.dll")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="CapacityConnect",
    console=True,      # set False to hide console
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
)
