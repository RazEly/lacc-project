"""Pipeline driver — full end-to-end run.

Per LM: one DAPT run per domain, then the mixed-model sweep for the reading-time
measure (TFT). Every surprisal source (baseline, aligned, prompted, …) enters as
a single plain main effect and is scored against the SAME shared no-surprisal
baseline (paper Eq. 2/4/5): LL, ΔLL, χ², LRT p, AIC. A second per-LM table
reports Vuong tests of the baseline-surprisal model against each
reader-conditioned arm (aligned per checkpoint, prompted, neutral prompt).
Along the way: figures (viz.md) — perplexity grid, example-sentence surprisal,
mean surprisal over steps, ΔLL curves, RT–surprisal.

    python -m src.main
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.analysis import model_comparison as mc
from src.analysis import viz
from src.config import WORD_KEY
from src.features import dataset as ds
from src.features import potec
from src.features import priors as pr
from src.features import surprisal as su
from src.modeling import finetune as ft
from src.modeling import lm

# reading-time measure (TFT); the effect is expected on the late measure.
MEASURE = "TFT"
SLIM_MODELS = ("baseline", "prompted", "prompt_neutral", "aligned")
# every checkpoint: indices 1..N pair to finetune's DAPT_CHECKPOINT_STEPS (0 = baseline).
SLIM_INDICES = list(range(1, len(ft.DAPT_CHECKPOINT_STEPS) + 1))

# Decoder LMs the pipeline runs over (slug -> HF repo), then compares — defined
# where they are trained (finetune). DAPT checkpoints are expected to already
# live in artifacts/; train them with `python -m src.modeling.finetune`.
MODELS = ft.MODELS
# Context budget (tokens) shared by every prompt condition.
PRIOR_MAX_TOKENS = 256

# Example sentence for the fig-2 surprisal walk-through: (text_id, sent_index).
# Baseline GPT-2 surprisal + Δ over checkpoints/prompts on one PoTeC sentence.
EXAMPLE_SENTENCE = ("b0", 1)
# The LM whose baseline drives the single-model figures (fig 2 + fig 5).
FIGURE_LM = "german-gpt2"


def dapt_manifest(name: str, domain: str) -> pd.DataFrame:
    """Load one finished DAPT run from ``artifacts/``.

    The pipeline assumes the models were already fine-tuned; train them with
    ``python -m src.modeling.finetune``.
    """
    out_dir = ft.run_dir_for(name, domain)
    cached = ft.load_cached_run(out_dir)
    if cached is None:
        raise FileNotFoundError(
            f"No DAPT run at {out_dir}. Train first: python -m src.modeling.finetune"
        )
    print(f"  [{domain}] {len(cached)} checkpoints in {out_dir}")
    return cached


def compute_model_surprisal(slug: str, name: str, words, priors: dict) -> dict:
    """Phase 1 — all surprisal for one model. No linear-model fitting here."""
    print(f"\n=== surprisal: {slug} ({name}) ===")

    print("Step 2 — load LM")
    model, tok = lm.load_causal_lm(name)

    # Prompted arm: prior-reading passage + native document boundary, truncated to
    # a shared context budget. physics/biology = domain priors (reader-aligned per
    # reader downstream); neutral = off-domain scientific control (same corpus,
    # same register, no domain content). Each arg is a LIST of N_PRIOR_PASSAGES
    # priors — compute_surprisal averages per-word surprisal across them, so the
    # stored column is the mean over priors.
    def _prior_surp(prior_texts: list[str], col: str) -> pd.DataFrame:
        return su.compute_surprisal(
            words,
            model,
            tok,
            prompt=prior_texts,
            max_prompt_tokens=PRIOR_MAX_TOKENS,
        ).rename(columns={"surprisal": col})

    prompt_surp = (
        _prior_surp(priors["physics"], "s_prompt_physics")
        .merge(_prior_surp(priors["biology"], "s_prompt_biology"), on=WORD_KEY)
        .merge(_prior_surp(priors["neutral"], "s_prompt_neutral"), on=WORD_KEY)
    )

    # Step 4 — DAPT checkpoints, one run per domain (loaded from artifacts/)
    print("Step 4 — DAPT checkpoints")

    manifest = pd.concat(
        [dapt_manifest(name, domain) for domain in config.DOMAINS],
        ignore_index=True,
    )

    surp_versions = su.recompute_surprisal_over_checkpoints(words, manifest).drop(
        columns=["checkpoint"]
    )

    return {
        "slug": slug,
        "prompt_surp": prompt_surp,
        "surp_versions": surp_versions,
        "manifest": manifest,
    }


def fit_model(
    bundle: dict, rm, measure: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phase 2 — mixed-model comparison for one model × measure.

    Returns ``(results, vuong, reader_ll)``, all tagged with ``model_lm`` /
    ``measure``.
    """
    slug = bundle["slug"]
    print(f"\n=== fit: {slug} / {measure} ===")

    # drop the neutral-prompt arm when the cache lacks it (older surprisal.csv).
    has_neutral = "s_prompt_neutral" in bundle["prompt_surp"].columns
    models = tuple(m for m in SLIM_MODELS if has_neutral or m != "prompt_neutral")

    # Model comparison — every source scored against the shared no-surprisal
    # baseline (LRT + AIC), plus Vuong tests of the baseline-surprisal model
    # vs the reader-conditioned arms. All checkpoints.
    cmp, vuong, reader_ll = mc.model_comparison_over_steps(
        bundle["surp_versions"],
        rm,
        bundle["prompt_surp"],
        measure=measure,
        models=models,
        indices=SLIM_INDICES,
    )
    for df in (cmp, vuong, reader_ll):
        df.insert(0, "model_lm", slug)
        df.insert(1, "measure", measure)
    print(cmp.to_string(index=False))
    print("\nVuong — baseline+surprisal vs reader-conditioned arms:")
    print(vuong.to_string(index=False))
    return cmp, vuong, reader_ll


def main() -> None:
    print("Step 1 — load PoTeC")
    rm_raw = potec.load_reading_measures()
    words = potec.load_word_features()
    print(f"  raw_rows={len(rm_raw)}  words={len(words)}")

    # Phase 1 — surprisal. If the cache exists, reload it straight from CSV
    # (no model load, no finetune) and go to fitting; trust its contents. Otherwise
    # compute every model, then write the cache once at the end (a crash mid-run
    # loses all of it — accepted trade-off for the simpler cache).
    bundles = su.load_cache(list(MODELS))
    if bundles is not None:
        print(f"Surprisal cache hit ({config.SURPRISAL_CACHE_PATH}) — skipping compute")
    else:
        priors = pr.load_prior_passages()
        print(
            f"  priors: {config.N_PRIOR_PASSAGES}× physics/biology from held-out "
            f"german-commons + {config.N_PRIOR_PASSAGES}× neutral (off-domain "
            f"pool), averaged per word (budget {PRIOR_MAX_TOKENS} tokens)"
        )
        bundles = [
            compute_model_surprisal(slug, name, words, priors)
            for slug, name in MODELS.items()
        ]
        su.save_cache(bundles)
        print(f"  wrote surprisal cache -> {config.SURPRISAL_CACHE_PATH}")

    # Surprisal-only figures — perplexity, example sentence, mean surprisal.
    # Perplexity needs the training manifests (None on a cache reload -> skipped).
    print("\nStep 4b — surprisal figures -> figures/")
    manifests = [
        b["manifest"].assign(model=b["slug"])
        for b in bundles
        if b.get("manifest") is not None
    ]
    if manifests:
        # ``perplexity`` is the run's OWN held-out split; ``perplexity_<domain>``
        # (newer manifests) is the other domain's held-out split. Melt to long
        # eval_domain form — old cached manifests lack the cross columns and
        # plot in-domain lines only (retrain to add the dotted lines).
        m = pd.concat(manifests, ignore_index=True).rename(
            columns={"domain": "ft_domain"}
        )
        long = [m.assign(eval_domain=m["ft_domain"])]
        for d in config.DOMAINS:
            col = f"perplexity_{d}"
            if col in m.columns:
                long.append(
                    m.assign(eval_domain=d, perplexity=m[col]).dropna(
                        subset=["perplexity"]
                    )
                )
        # PoTeC-stimuli lines (perplexity_grid ``show_stimuli`` toggles them,
        # default on): per-checkpoint token perplexity on the physics/biology
        # stimulus texts, derived from the cached per-word surprisal — only the
        # tokenizer is loaded, no extra forward passes.
        stimuli = pd.concat(
            [
                su.stimuli_perplexity(
                    b["surp_versions"], words, lm.load_tokenizer(MODELS[b["slug"]])
                ).assign(model=b["slug"])
                for b in bundles
            ],
            ignore_index=True,
        ).rename(columns={"domain": "ft_domain"})
        viz.perplexity_grid(pd.concat(long, ignore_index=True), stimuli=stimuli)

    for b in bundles:
        for expert_only in (False, True):
            viz.mean_surprisal_over_steps(
                b["surp_versions"], b["prompt_surp"], words, b["slug"],
                expert_terms_only=expert_only,
            )

    fig_lm = next((b for b in bundles if b["slug"] == FIGURE_LM), bundles[0])
    text_id, sent_index = EXAMPLE_SENTENCE
    viz.sentence_surprisal_example(
        fig_lm["surp_versions"],
        fig_lm["prompt_surp"],
        words,
        text_id,
        sent_index,
        slug=fig_lm["slug"],
    )

    # Phase 2 — mixed models per measure (early + late) per LM.
    all_cmp, all_vuong = [], []
    rm_by_measure = {}
    rm = ds.clean_reading_times(rm_raw, MEASURE)
    rm_by_measure[MEASURE] = rm
    print(f"\nStep 5 — {MEASURE}: cleaned={len(rm)} ({len(rm) / len(rm_raw):.1%})")
    for b in bundles:
        cmp, vuong, _ = fit_model(b, rm, MEASURE)
        all_cmp.append(cmp)
        all_vuong.append(vuong)

    results = pd.concat(all_cmp, ignore_index=True)
    results_path = config.PROJECT_ROOT / "results_slim.csv"
    results.to_csv(results_path, index=False)
    vuong = pd.concat(all_vuong, ignore_index=True)
    vuong_path = config.PROJECT_ROOT / "results_vuong.csv"
    vuong.to_csv(vuong_path, index=False)
    print(f"\n  wrote {results_path.name}, {vuong_path.name}")

    # Fit-dependent figures — ΔLL curves (all LMs) + RT vs baseline surprisal
    # (baseline of FIGURE_LM, first measure only).
    print("\nStep 5b — fit figures -> figures/")
    viz.delta_ll_curves(results)
    baseline_surp = fig_lm["surp_versions"].query("index == 0")
    viz.rt_vs_surprisal(
        rm_by_measure[MEASURE], baseline_surp, measure=MEASURE, slug=fig_lm["slug"]
    )


if __name__ == "__main__":
    main()
