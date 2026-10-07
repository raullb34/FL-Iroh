"""Assemble fl-iroh-cas-dc_R1.tex from the original header/bibliography and the R1 sections."""
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
orig = (root / "fl-iroh-cas-dc.tex").read_text(encoding="utf-8").split("\n")


def r(name: str) -> str:
    return (here / name).read_text(encoding="utf-8")


i_cortext = next(i for i, l in enumerate(orig) if l.startswith(r"\cortext"))
i_credit = next(i for i, l in enumerate(orig) if l.startswith(r"% ---- Elsevier back matter"))
i_endbib = next(i for i, l in enumerate(orig) if l.startswith(r"\end{thebibliography}"))

head = "\n".join(orig[: i_cortext + 1])
head = head.replace(
    r"\usepackage{pgfplots}",
    r"\usepackage{pgfplots}" + "\n"
    + r"\newcommand{\TBD}[1]{\textcolor{red}{[TBD#1]}}  % R1: pending values; remove before submission",
)
back = "\n".join(orig[i_credit:i_endbib])
tail = "\n".join(orig[i_endbib:])

s3 = r("s3_system.tex").replace(
    r"and Algorithm~\ref{alg:round} the aggregator's logic",
    r"and Table~\ref{alg:round} the aggregator's logic (Algorithm~1)",
)
s4a = (r("s4_eval_a.tex")
       .replace("C1--C5 & 2025 cross-ISP campaign", "C1--C5 & Earlier cross-ISP campaign")
       .replace(r"E7 & (reserved for the operational comparison, \S\ref{sec:flower}) & --- & --- \\",
                r"E7 & Operational comparison with Flower + Tailscale (\S\ref{sec:flower}) & HPC & in-process \\"))
s4b = r("s4_eval_b.tex").replace(
    "All 260 rounds completed in every condition.",
    "All 320 rounds of the 16 completed conditions finished successfully (13 conditions are shown).",
)
for a, b in ((s3, "alg"), (s4a, "C1"), (s4b, "320")):
    pass

doc = "\n".join([
    head, "", r("s1_front.tex"), r("s2_related.tex"), s3, s4a, s4b,
    r("s5_flower_discussion.tex"), r("s6_conclusion.tex"), "",
    back, r("bib_new.tex"), r("bib_agent.tex"), tail,
])
assert "All 320 rounds" in doc and "Earlier cross-ISP" in doc and "Table~\\ref{alg:round}" in doc
(root / "fl-iroh-cas-dc_R1.tex").write_text(doc, encoding="utf-8", newline="\n")
print("lines:", len(doc.split("\n")))
