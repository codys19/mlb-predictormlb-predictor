"""Run from C:\\MLB: python fix_gitignore2.py"""
path = '.gitignore'

# Read with UTF-16 (the actual encoding)
try:
    content = open(path, 'r', encoding='utf-16').read()
    print("Read as UTF-16")
except:
    content = open(path, 'r', encoding='utf-8').read()
    print("Read as UTF-8")

lines = content.splitlines()
keep = []
for l in lines:
    stripped = l.strip().replace('\x00', '')
    if not stripped:
        continue
    if 'bullpen_usage' in stripped:
        print(f"  Removing: {repr(stripped)}")
        continue
    if 'mlb 2 github' in stripped and stripped != 'mlb 2 github.txt':
        print(f"  Removing malformed: {repr(stripped)}")
        continue
    keep.append(stripped)

# Write clean UTF-8 with no BOM
result = '\n'.join(keep) + '\n'
open(path, 'w', encoding='utf-8', newline='\n').write(result)

print("\nFinal .gitignore:")
print(open(path, 'r', encoding='utf-8').read())
