"""Standalone fig1 (perplexity grid) — reads cached DAPT manifests, no model load.

Replicates the perplexity-figure block of src.main without the pipeline. Run:
    pixi run python -m scripts.plot_perplexity
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis import viz
from src.config import MODELS
from src.modeling import finetune as ft


def main() -> None:
    manifests = []
    for slug, name in MODELS.items():
        for domain in config.DOMAINS:
            m = ft.load_cached_run(ft.run_dir_for(name, domain))
            if m is None:
                raise FileNotFoundError(f"no cached run for {slug}/{domain}")
            manifests.append(m.assign(model=slug))

    m = pd.concat(manifests, ignore_index=True).rename(columns={"domain": "ft_domain"})
    long = [m.assign(eval_domain=m["ft_domain"])]
    for d in config.DOMAINS:
        col = f"perplexity_{d}"
        if col in m.columns:
            long.append(
                m.assign(eval_domain=d, perplexity=m[col]).dropna(subset=["perplexity"])
            )

    viz.perplexity_grid(
        pd.concat(long, ignore_index=True), stimuli=None, show_stimuli=False
    )
    print(f"wrote {viz.FIGURES_DIR / 'fig1_perplexity.png'}")


if __name__ == "__main__":
    main()
