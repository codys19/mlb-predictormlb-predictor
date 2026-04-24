"""Run from C:\\MLB: python data/fix_gitignore.py"""
path = '.gitignore'
lines = open(path, 'r', encoding='utf-8').read().splitlines()

print("Before:")
for l in lines:
    print(f"  {repr(l)}")

keep = []
for l in lines:
    stripped = l.strip()
    if 'bullpen_usage' in stripped:
        print(f"  Removing: {repr(l)}")
        continue
    if 'mlb 2 github' in stripped and l != 'mlb 2 github.txt':
        print(f"  Removing malformed: {repr(l)}")
        continue
    keep.append(l)

result = '\n'.join(keep) + '\n'
open(path, 'w', encoding='utf-8').write(result)

print("\nAfter:")
print(open(path).read())
