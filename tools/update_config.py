import re
from pathlib import Path
p = Path("config.py")
s = p.read_text(encoding="utf-8")
rules = [(r'("pretrain_epochs"\s*:\s*)[0-9]+', r'\g<1>50'),
         (r'("temperature"\s*:\s*)[0-9.]+', r'\g<1>0.5')]
for pat, rep in rules:
    if len(re.findall(pat, s)) != 1:
        raise SystemExit(f"Stopped - setting not found exactly once ({pat}). config.py NOT changed.")
for pat, rep in rules:
    s = re.sub(pat, rep, s)
p.write_text(s, encoding="utf-8")
for line in s.splitlines():
    if "pretrain_epochs" in line or '"temperature"' in line:
        print("NOW:", line.strip())
