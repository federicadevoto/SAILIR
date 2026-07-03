"""
Build a Python dict (sector_id -> [master integral tuples]) from
Kira's masters file for the TB (HexaBox) topology.

Output goes to TB_masters_dict.py for import by SAILIR data-gen.
"""
import re
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
masters = []
with open(HERE / "masters") as f:
    for ln in f:
        m = re.match(r'TB\[([^\]]+)\]', ln.strip())
        if not m:
            continue
        v = tuple(int(x.strip()) for x in m.group(1).split(','))
        assert len(v) == 11
        masters.append(v)

def sector_of(v):
    return sum((1 << i) for i, a in enumerate(v) if a > 0)

by_sec = defaultdict(list)
for v in masters:
    by_sec[sector_of(v)].append(v)

# Write dict source file
out = HERE / "TB_masters_dict.py"
with open(out, "w") as f:
    f.write('"""\n')
    f.write("HexaBox (TB) master integral basis from Kira reduction\n")
    f.write("(target TB[2,1,1,1,1,1,1,1,0,0,0], sector 255, r=10, s=4,\n")
    f.write("run_initiate+run_triangular+run_back_substitution; 73 masters total).\n")
    f.write("\n")
    f.write("Same schema as TA_masters_dict.py:\n")
    f.write("  sector_id -> list of master integral tuples (length-11).\n")
    f.write('"""\n\n')
    f.write("TB_MASTERS = {\n")
    for sec in sorted(by_sec):
        items = by_sec[sec]
        f.write(f"    {sec}: [\n")
        for m in items:
            f.write(f"        {m},\n")
        f.write("    ],\n")
    f.write("}\n\n")
    f.write("# 8-denominator + 3-ISP convention: bits 0..7 of sector_id\n")
    f.write("# correspond to D1..D8 (denominators); positions 8..10 are\n")
    f.write("# D9, D10, D11 (always ISPs, never positive in masters).\n")
    f.write("ISP_POSITIONS = (8, 9, 10)\n")
    f.write("N_INDICES = 11\n")
    f.write("N_DENOMINATORS = 8\n")

print(f"wrote {out}")
print(f"sectors with masters: {len(by_sec)}")
print(f"total masters: {sum(len(v) for v in by_sec.values())}")
