"""Run this from C:\\MLB to patch data/backtest.py"""
txt = open('data/backtest.py', 'r', encoding='utf-8').read()

old = 'test = test.dropna(subset=FEATURE_COLS + ["home_team_won"])'
new = (
    'required = [c for c in FEATURE_COLS if not any(\n'
    '        c.startswith(p) for p in ("home_starter_","away_starter_","starter_","has_individual_")\n'
    '    )]\n'
    '    test = test.dropna(subset=required + ["home_team_won"])'
)

if old in txt:
    txt = txt.replace(old, new)
    open('data/backtest.py', 'w', encoding='utf-8').write(txt)
    print("✅ Patched successfully")
elif 'required + ["home_team_won"]' in txt:
    print("✅ Already patched")
else:
    print("❌ Could not find target line — backtest.py may be corrupted")
    print("   Lines containing 'dropna':")
    for i, line in enumerate(txt.splitlines(), 1):
        if 'dropna' in line:
            print(f"   Line {i}: {line}")
