#!/usr/bin/env bash
# Compile project.tex: pdflatex -> biber -> pdflatex x2 (resolves citations + cross-refs).
# Usage: ./compile.sh [texfile-basename]   (default: project)
set -euo pipefail

cd "$(dirname "$0")"
DOC="${1:-project}"

pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" >/dev/null
biber "$DOC" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" | grep -E "^!|Warning|Output written" || true

echo "OK: $DOC.pdf"
