# lang-comp-cog-domain

Domain labelling of german-commons + PoTeC reading-time analysis.

## Setup

Python deps are managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### R backend for the mixed-effects models

The surprisal ↔ reading-time analysis (`src/analysis/model_comparison.py`,
`src/analysis/stats.py`) fits crossed-random-effect models with **lme4** through
[`pymer4`](https://eshinjolly.com/pymer4/), which drives R via `rpy2`. This
mirrors Škrjanec & Demberg — the exact `(1|reader_id) + (1 + is_expert|word_id)`
structure and Satterthwaite p-values are not available in pure-Python mixed-model
libraries.

`uv` installs the Python side (`pymer4`, `rpy2`); the R side is provided by a
self-contained **conda** environment (precompiled, no system R / Fortran / root):

```bash
conda create -n renv -c conda-forge -y r-base=4.4 r-lme4 r-lmertest
```

`rpy2` must load *this* R (it is built against it), not any system R. Point it
there with a `.env` (copy `.env.example`, fix the path to your conda env) and run
the pipeline with `--env-file`:

```bash
cp .env.example .env            # then edit R_HOME / LD_LIBRARY_PATH if needed
uv run --env-file .env python -m src.main
uv run --env-file .env pytest
```

Version pins that must hold (tied to the classic pymer4 API):
`pymer4 < 0.9` (0.9 is a polars/easystats rewrite with a different API and a large
R-package chain), `rpy2 < 3.6` (3.6 dropped `NULL.rx2`, which pymer4 0.8.x needs —
hence R 4.4, the newest R that rpy2 3.5 supports), and `pandas < 3` (pymer4 0.8.x
still calls the removed `DataFrame.applymap`).
