# lang-comp-cog-domain

Domain labelling of german-commons + PoTeC reading-time analysis.

## Setup

Python deps are managed by [uv](https://docs.astral.sh/uv/); run everything via `uv run`:

```bash
uv sync
uv run python -m src.main
uv run pytest
```

### System prerequisites (mixed-effects models)

The surprisal ↔ reading-time analysis (`src/analysis/model_comparison.py`,
`src/analysis/stats.py`) fits crossed-random-effect models with **lme4** through
[`pymer4`](https://eshinjolly.com/pymer4/), which drives R via `rpy2`. This
mirrors the methodology of Škrjanec & Demberg — the exact `(1|reader_id) + (1 +
is_expert|word_id)` structure and Satterthwaite p-values are not available in
pure-Python mixed-model libraries.

`uv` installs the Python side (`pymer4`, `rpy2`), but **R and the R packages are
system dependencies** you must install out of band on each machine:

```bash
# system R + a Fortran compiler (lme4 -> minqa/nloptr need Fortran)
sudo pacman -S --needed r gcc-fortran          # Arch
# sudo apt install r-base gfortran             # Debian/Ubuntu

# lme4 + lmerTest into the R user library (no root needed)
R --slave -e 'p <- Sys.getenv("R_LIBS_USER"); dir.create(p, recursive=TRUE, showWarnings=FALSE); install.packages(c("lme4","lmerTest"), repos="https://cloud.r-project.org", lib=p)'
```

Version pins that matter: `pymer4 < 0.9` (0.9 is a polars-based rewrite with a
different API) and `pandas < 3` (`pymer4` 0.8.x still calls the removed
`DataFrame.applymap`).
