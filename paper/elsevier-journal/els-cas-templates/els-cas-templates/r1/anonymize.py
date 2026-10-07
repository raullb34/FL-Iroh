"""Derive the double-blind version of the R1 manuscript (same transforms as the original anonymous file)."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
s = (root / "fl-iroh-cas-dc_R1.tex").read_text(encoding="utf-8")

s = s.replace("%% FL-Iroh -- Elsevier CAS (cas-dc) double-column manuscript",
              "%% FL-Iroh -- Elsevier CAS (cas-dc) double-column manuscript (ANONYMOUS FOR BLIND REVIEW)", 1)
s = re.sub(r"\\shortauthors\{[^\n]*\}", lambda m: r"\shortauthors{Anonymous}", s, count=1)
a = s.index("% ---- Authors")
b = s.index(r"\cortext[1]{Corresponding author}") + len(r"\cortext[1]{Corresponding author}")
s = s[:a] + "% ---- Authors (Anonymous for blind review) --------------------------------\n\\author{Anonymous Submission}" + s[b:]

s = s.replace(r"Code, per-run results and analysis scripts are available at \url{https://github.com/raullb34/FL-Iroh}.",
              "Code, per-run results and analysis scripts are available at the anonymized project repository (URL withheld for double-blind review).")
s = s.replace(r"are available at \url{https://github.com/raullb34/FL-Iroh}.",
              "are available at an anonymized project repository (URL withheld for double-blind review; will be disclosed upon acceptance).")
a = s.index(r"\section*{CRediT authorship contribution statement}")
b = s.index(r"\section*{Declaration of competing interest}")
s = s[:a] + s[b:]
s = re.sub(r"\\section\*\{Acknowledgements\}\n[^\n]*\n", "", s)
s = s.replace("% ---- Elsevier back matter ------------------------------------------------",
              "% ---- Elsevier back matter (anonymized) ------------------------------------")

leaks = [w for w in ["raullb34", "L\\'{o}pez", "L\\'opez", "Alonso", "Prieto", "Pinto", "OSTARA", "BISITE",
                     "usal.es", "air-institute"] if w in s]
(root / "fl-iroh-cas-dc_R1_anonymous.tex").write_text(s, encoding="utf-8", newline="\n")
print("possible identity leaks:", leaks)
