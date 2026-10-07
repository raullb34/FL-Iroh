"""Static checks for the R1 manuscript: citations, refs, environment balance, pending values."""
import re
import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / (sys.argv[1] if len(sys.argv) > 1 else "fl-iroh-cas-dc_R1.tex")
t = path.read_text(encoding="utf-8")
body = re.sub(r"(?m)^%.*$", "", t)
cites = {k.strip() for g in re.findall(r"\\cite[tp]?\{([^}]*)\}", body) for k in g.split(",")}
bibs = set(re.findall(r"\\bibitem\{([^}]*)\}", body))
labels = set(re.findall(r"\\label\{([^}]*)\}", body))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", body))
print("missing bibitems :", sorted(cites - bibs))
print("uncited bibitems :", sorted(bibs - cites))
print("undefined refs   :", sorted(refs - labels))
dup = [l for l in labels if body.count("\\label{%s}" % l) > 1]
print("duplicate labels :", dup)
for env in ["table", "table*", "figure", "tabular", "itemize", "enumerate", "tikzpicture", "equation"]:
    b, e = body.count("\\begin{%s}" % env), body.count("\\end{%s}" % env)
    if b != e:
        print("UNBALANCED", env, b, e)
print("braces balance   :", body.count("{") - body.count("}"))
print("pending \\TBD     :", body.count("\\TBD{"))
for w in ["privacy-preserving", "Raspberry Pi~4", "Pi 4B", "guarantee", "0\\% relay"]:
    n = body.count(w)
    if n:
        print(f"check wording '{w}':", n)
