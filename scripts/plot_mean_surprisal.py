"""Standalone fig3 (mean surprisal over steps) — reads surprisal cache, no model load.

    pixi run python -m scripts.plot_mean_surprisal
"""

from __future__ import annotations

from src.analysis import viz
from src.config import MODELS
from src.features import potec
from src.features import surprisal as su


def main() -> None:
    bundles = su.load_cache(list(MODELS))
    if bundles is None:
        raise FileNotFoundError("no surprisal cache")
    words = potec.load_word_features()
    for expert_only in (False, True):
        viz.mean_surprisal_over_steps(bundles, words, expert_terms_only=expert_only)
    print(f"wrote fig3 -> {viz.FIGURES_DIR}")


if __name__ == "__main__":
    main()
