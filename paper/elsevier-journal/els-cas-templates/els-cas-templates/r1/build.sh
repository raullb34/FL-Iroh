#!/usr/bin/env bash
# Local build of the R1 manuscript (and optional latexdiff) in a scratch copy.
# The CAS template expects thumbnails/cas-*.jpeg (shipped by Overleaf/Elsevier,
# not in this repo); small placeholder icons are generated for local builds only.
#   usage: r1/build.sh [diff]
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
IMG="$(cd "$SRC/../../../img" && pwd)"
B=/tmp/r1src
rm -rf "$B"; mkdir -p "$B/thumbnails" "$B/img"
cp "$SRC"/*.tex "$SRC"/*.cls "$SRC"/*.sty "$SRC"/*.bst "$B"/
cp "$IMG"/* "$B/img/"
PY="$SRC/../../../../.venv/bin/python"
"$PY" - "$B/thumbnails" <<'PY'
import sys
from PIL import Image
for n in ["email", "facebook", "gplus", "linkedin", "twitter", "url"]:
    Image.new("RGB", (16, 16), (90, 90, 90)).save(f"{sys.argv[1]}/cas-{n}.jpeg", "JPEG")
PY
cd "$B"
sed -i 's#\\graphicspath{{../../../img/}}#\\graphicspath{{img/}}#' fl-iroh-cas-dc.tex fl-iroh-cas-dc_anonymous.tex fl-iroh-cas-dc_R1.tex fl-iroh-cas-dc_R1_anonymous.tex
if [[ "${1:-}" == "diff" ]]; then
    latexdiff --type=UNDERLINE --math-markup=whole --exclude-textcmd=textbf --config="PICTUREENV=(?:picture|DIFnomarkup|tabular|tikzpicture)[wd*@]*" fl-iroh-cas-dc_anonymous.tex fl-iroh-cas-dc_R1_anonymous.tex > fl-iroh-cas-dc_R1_diff.tex
    targets=(fl-iroh-cas-dc_R1.tex fl-iroh-cas-dc_R1_anonymous.tex fl-iroh-cas-dc_R1_diff.tex response_to_reviewers.tex response_to_reviewers_es.tex)
else
    targets=(fl-iroh-cas-dc_R1.tex response_to_reviewers.tex)
fi
for t in "${targets[@]}"; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$t" > "${t%.tex}.latexmk.out" 2>&1 \
        && echo "OK   $t ($(grep -o 'Output written.*' "${t%.tex}.log"))" \
        || { echo "FAIL $t"; grep -n -A3 '^!' "${t%.tex}.log" | head -20; }
done
