"""Replace provisional table/section numbers in the response letter with the R1 numbering."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tex = (root / "fl-iroh-cas-dc_R1.tex").read_text(encoding="utf-8")

# Float numbering follows source order of \begin{table}/\begin{table*} (same counter).
tables = re.findall(r"\\begin\{table\*?\}.*?\\label\{([^}]*)\}", tex, flags=re.S)
tnum = {lab: i + 1 for i, lab in enumerate(tables)}
print({k: v for k, v in tnum.items()})

p = root / "response_to_reviewers.tex"
s = p.read_text(encoding="utf-8")
T = lambda lab: str(tnum[lab])
reps = [
    (r"(Table~\ref{x}~4)", f"(Table~{T('tab:networks')})"),
    (r"\S4.1 and Table~4 (networks and RFC~4787 classification); \S4.4/E3 rewritten, with a new Table~7 that replaces the former transfer-level table; \S4.10/E8;",
     rf"\S4.1 and Table~{T('tab:networks')} (networks and RFC~4787 classification); \S4.4/E3 rewritten, with Table~{T('tab:e3est')} (strictly classified cold starts) replacing the former transfer-level table; \S4.8/E8;"),
    (r"(Table~\ref{x}~7)", f"(Table~{T('tab:e3est')})"),
    (r"(Table~\ref{x}~10)", f"(Table~{T('tab:e8')})"),
    (r"New \S4.10 (E8) and Table~10;", rf"New \S4.8 (E8) and Table~{T('tab:e8')};"),
    (r"We now state explicitly that Table~8 is an architectural comparison",
     rf"We now state explicitly that Table~8 of the original submission (now Table~{T('tab:e7')}) is an architectural comparison"),
    (r"We also updated Table~8 with", rf"We also updated Table~{T('tab:e7')} with"),
    (r"\S4.13; Table~8; Table~\ref{x}~16 (roadmap).",
     rf"\S4.12; Tables~{T('tab:e7')}--{T('tab:e7acc')}; Table~{T('tab:roadmap')} (roadmap)."),
    (r"(E11, Table~14)", f"(E11, Table~{T('tab:e11')})"),
    (r"New \S4.13 (E11) and Table~14;", rf"New \S4.11 (E11) and Table~{T('tab:e11')};"),
    (r"\emph{estimated* with", r"\emph{estimated} with"),
    (r"new \S4.13 (E11); Limitations.", r"new \S4.11 (E11); Limitations."),
    (r"comparison table (Table~1)", f"comparison table (Table~{T('tab:related')})"),
    (r"\S2.2; Table~1.", rf"\S2.2; Table~{T('tab:related')}."),
    (r"\S4.5 (E4 scope); new \S4.10 (E8); \S4.13 (E11);", r"\S4.5 (E4 scope); new \S4.8 (E8); \S4.11 (E11);"),
    (r"(E9, Table~11)", f"(E9, Table~{T('tab:e9')})"),
    (r"New \S4.11 (E9) and Table~11.", rf"New \S4.9 (E9) and Table~{T('tab:e9')}."),
    (r"(Table~\ref{x}~9)", f"(Table~{T('tab:e5')})"),
    (r"\S4.6 (E5) and Table~9 replaced.", rf"\S4.6 (E5) and Table~{T('tab:e5')} replaced."),
    (r"Results (Table~12):", f"Results (Table~{T('tab:e10')}):"),
    (r"\S4.12 (E10) and Table~12; \S5.4; Table~8.", rf"\S4.10 (E10) and Table~{T('tab:e10')}; \S5.4; Table~{T('tab:e7')}."),
    (r"\S1; \S5.4; Conclusion; Table~16.", rf"\S1; \S5.4; Conclusion; Table~{T('tab:roadmap')}."),
    (r"Table~\ref{x}~13 reports", f"Table~{T('tab:e6')} reports"),
    (r"\S4.7 (E6) and Table~13.", rf"\S4.7 (E6) and Table~{T('tab:e6')}."),
    (r"\S1; \S3.4; new \S5.3; Table~16.", rf"\S1; \S3.4; new \S5.3; Table~{T('tab:roadmap')}."),
    (r"Added (Table~1, \S2.8).", f"Added (Table~{T('tab:related')}, \\S2.8)."),
    (r"New \S2.8 and Table~1.", rf"New \S2.8 and Table~{T('tab:related')}."),
    (r"\S2; Table~1; References.", rf"\S2; Table~{T('tab:related')}; References."),
    (r"Added as Table~0 (``Abbreviations'')", f"Added as Table~{T('tab:abbrev')} (``Abbreviations'')"),
    (r"(Fig.~4) and the aggregator's round algorithm (Algorithm~1);",
     f"(Fig.~4) and the aggregator's round algorithm (Algorithm~1, Table~{T('alg:round')});"),
    (r"\S3 (Tables~2--3, Fig.~4, Algorithm~1, Eq.~2).",
     rf"\S3 (Tables~{T('tab:components')}--{T('alg:round')}, Fig.~4, Eq.~2)."),
    (r"research roadmap (Table~16)", f"research roadmap (Table~{T('tab:roadmap')})"),
    (r"\S6 and Table~16.", rf"\S6 and Table~{T('tab:roadmap')}."),
]
missing = []
for a, b in reps:
    if a not in s:
        missing.append(a)
    s = s.replace(a, b)
left = re.findall(r"\\ref\{x\}", s)
p.write_text(s, encoding="utf-8", newline="\n")
print("not found:", missing)
print("remaining placeholders:", len(left))
