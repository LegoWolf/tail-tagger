# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['e621_service.py'],
    pathex=[],
    binaries=[
        ("exiv2.exe", "."),
        ("exiv2.dll", "."),
    ],
    datas=[
        ("e621_classifier.toml", "."),
        ("classifiers/JTP-3/jtp-3-hydra.safetensors", "classifiers/JTP-3"),
    ],
    hiddenimports=[
        "win32timezone"
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='e621_service',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='e621_service',
)
