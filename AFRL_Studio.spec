# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['MainGui.py'],
    pathex=[
        '.',
        'fragment_extractor',
        'feature_extractor',
        'feature_extractor/NIST',
        'feature_extractor/HEDGE_specials',
        'feature_extractor/BLFCP_specials',
        'feature_extractor/SOTA_specials',
        'feature_extractor/AFRL_specials',
        'feature_extractor/EnCoD_specials',
        'feature_extractor/Triumvir_specials',
        'combine_csv',
    ],
    binaries=[],
    datas=[('icons', 'icons'), ('fragment_extractor', 'fragment_extractor'), ('feature_extractor', 'feature_extractor'), ('combine_csv', 'combine_csv')],
    hiddenimports=[
        'nn_utils',
        'EnCoD_fspecials',
        'BFLCP_dnn',
        'Triumvir_ensemble',
        'scipy.optimize',
        'scipy.fft',
        'scipy.stats',
        'scipy.special',
        'zlib',
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
    name='AFRL_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons/main_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AFRL_Studio',
)
