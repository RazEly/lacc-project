# lang-comp-cog-domain

Term-targeted Wikipedia DAPT corpora + PoTeC reading-time analysis.

## Setup

The whole environment — Python deps **and** the R backend — is managed by a
single tool, [pixi](https://pixi.sh). One lockfile (`pixi.lock`) pins everything
for `linux-64` and `osx-arm64`, so a fresh machine is reproducible with no conda,
no `uv`, and no hand-set `R_HOME`.

```bash
# 1. install pixi (no root)
curl -fsSL https://pixi.sh/install.sh | bash   # then restart your shell

# 2. build the env from the lockfile (Python + R + lme4/lmerTest + easystats)
pixi install

# 3. fetch data + run
./init.sh                 # clone PoTeC, download eye-tracking data, scrape the domain corpora
pixi run main             # = python -m src.main
pixi run test             # = pytest
```

### R backend for the mixed-effects models

The surprisal ↔ reading-time analysis (`src/analysis/model_comparison.py`)
fits crossed-random-effect models with **lme4** through
[`pymer4`](https://eshinjolly.com/pymer4/), which drives R via `rpy2`. This
mirrors Škrjanec & Demberg — the exact `(1|reader_id) + (1 + is_expert|word_id)`
structure and Satterthwaite p-values are not available in pure-Python mixed-model
libraries.

pixi supplies R and the full package chain pymer4 0.9 imports (`lme4`, `lmerTest`,
plus the easystats stack: `broom`, `broom.mixed`, `emmeans`, `insight`,
`parameters`, `performance`, `report`, `tibble`) from conda-forge — precompiled,
no system R / Fortran / root. `rpy2` comes from conda-forge too (its PyPI sdist
won't link against the conda R); inside the pixi env that R is first on `PATH`, so
`rpy2` auto-resolves `R_HOME` — no `.env` needed.
