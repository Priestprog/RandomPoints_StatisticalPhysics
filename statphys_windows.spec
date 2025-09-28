# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['tests/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('tests/strategies.py', '.'),
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.figure',
        'matplotlib.pyplot',
        'numpy',
        'scipy',
        'scipy.ndimage',
        'strategies',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='StatPhys',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Убираем консоль для Windows GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Можно добавить путь к .ico файлу
    version='version_info.txt'  # Можно добавить информацию о версии
)