"""Run from C:\\MLB: python data/add_retrain_step.py"""
import os

path = '.github/workflows/weekly_refresh.yml'
txt  = open(path, 'r', encoding='utf-8').read()

print("Current weekly_refresh.yml steps:")
for line in txt.splitlines():
    if '- name:' in line or 'run:' in line.lstrip():
        print(f"  {line}")

old = '''      - name: Commit updated game_results.csv
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add raw/game_results.csv
          git diff --staged --quiet || git commit -m "Weekly refresh: game_results.csv $(date +'%Y-%m-%d')"
          git push'''

new = '''      - name: Rebuild training features
        run: |
          python data/build_features.py

      - name: Retrain model
        run: |
          python data/train_model.py

      - name: Run backtest
        run: |
          python data/backtest.py

      - name: Commit updated files
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add raw/game_results.csv raw/model.json || true
          git add raw/backtest_results.csv raw/backtest_report.html raw/feature_importance.csv || true
          git diff --staged --quiet || git commit -m "Weekly refresh + retrain $(date +'%Y-%m-%d')"
          git push'''

if old in txt:
    open(path, 'w', encoding='utf-8').write(txt.replace(old, new))
    print('\n✅ Added retrain steps to weekly_refresh.yml')
else:
    print('\n❌ Could not find commit block — printing full file:')
    print(txt)
