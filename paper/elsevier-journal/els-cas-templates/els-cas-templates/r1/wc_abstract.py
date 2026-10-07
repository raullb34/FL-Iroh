import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def words(tex: str) -> int:
    a = tex[tex.index(r"\begin{abstract}") + len(r"\begin{abstract}"):tex.index(r"\end{abstract}")]
    a = re.sub(r"\\[a-zA-Z]+", " ", a)
    a = re.sub(r"[{}~$\\]", " ", a)
    return len(a.split())


print("R1 abstract words:", words((root / "r1" / "s1_front.tex").read_text(encoding="utf-8")))
print("original abstract words:", words((root / "fl-iroh-cas-dc.tex").read_text(encoding="utf-8")))
