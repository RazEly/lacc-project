"""Aligned-ATTENTION model comparison on PoTeC with full fine-tuned GPT-2.

Replicates the reader-aligned model comparison of ``src/main.py`` — which asks
whether *aligned surprisal* (physics-LM surprisal for physicists, biology-LM for
biologists) predicts reading time better than any single model — but swaps
surprisal for **raw attention** at the layer that best correlates with gaze (found
in the Mouratidi & Poesio replication and confirmed after the within-sentence
relative-normalization fix; see [[gaze-attention-relative-normalization]]).

Pipeline:
  1. **Full fine-tuning** (continued pre-training, ``lora=False``) of german-gpt2 on
     the physics and biology domains, separately, using Škrjanec et al.'s recipe
     (papers/07): batch 8, lr 1e-4, 16,384 steps, checkpoints at 4ⁿ steps — by
     step, not epoch.
  2. Raw attention from three models — the un-fine-tuned baseline, the physics-FT
     model, the biology-FT model — extracted per word per layer.
  3. The peak attention-vs-gaze layer L is chosen from the per-layer Spearman
     correlation (PCA gaze component, all readers).
  4. **Robust domain-attention vs expert reading-time analysis.** Raw aligned
     attention was a null (it's surface-dominated and shared across FT models), so
     this uses the stronger, residualized design (after Škrjanec et al.):
       (a) **specialized attention** = each FT model's layer-L attention residualized
           on the baseline (the domain-specific component, surface removed); compared
           as sources by ΔLL (general / physics / biology / aligned) per spec;
       (b) **reader-clustered Vuong** significance of aligned vs each single source;
       (c) PRIMARY: a paper-09 **three-way LRT** — does aligned-specialized attention
           add over general attention specifically for *experts on technical terms*?
       (d) model-free **ΔA-vs-Δ(expert speed-up) correlation** on technical terms,
           where the surface signal cancels on both sides.

GPU needed for step 1 (full FT of two models). Run from the project root:

    uv run python -m src.main_attention
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.analysis import correlation as co
from src.analysis import model_comparison as mc
from src.analysis import viz
from src.features import attention as at
from src.features import data
from src.features import reading_time as rt
from src.features import surprisal as su

MEASURE = "TFT"  # total fixation time == TRT (reading-time response)
NAME = config.MODELS["german-gpt2"]  # dbmdz/german-gpt2 (full FT target)
# Full fine-tuning (continued pre-training), NOT LoRA. Training recipe follows
# Škrjanec et al. (papers/07): batch 8, lr 1e-4, 100 warm-up steps, checkpoints at
# 4ⁿ steps — by STEP, not by epoch. NOTE: MAX_STEPS truncated to 4096 (a quarter of
# the paper's 16,384 → ~4× less GPU time); the final FT model is the 4096-step
# (~1-epoch) checkpoint. Deliberate deviation from the paper's 16,384-step recipe.
FINETUNE_LORA = False
DAPT_LR = 1e-4
BATCH_SIZE = 8
MAX_STEPS = 4096
WARMUP_STEPS = 100
PAPER_CHECKPOINT_STEPS = [4, 16, 64, 256, 1024, 4096]
# Attention sources compared (reader-aligned vs single models / baseline), exactly
# the main.py set minus the prompted variants (no prompts on the attention side).
ALIGNED_MODELS = ("baseline", "physics", "biology", "aligned")
WORD_KEY = ["text_id", "word_index_in_text"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"


def save_fig(ax, name: str) -> None:
    fig = ax.get_figure()
    fig.tight_layout()
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


def finetune_full(name: str) -> dict[str, str]:
    """Full fine-tune (continued pre-training) physics and biology; return checkpoints.

    Trains both domains for MAX_STEPS (4096 — a quarter of the paper's 16,384, to
    cut GPU time ~4×), checkpointing at the 4ⁿ schedule (by step, not epoch).
    Returns the baseline (step-0) checkpoint plus each domain's final (4096-step)
    fully-fine-tuned checkpoint. Requires a GPU.
    """
    from src.modeling import finetune as ft

    kw = dict(base_model=name, max_steps=MAX_STEPS,
              checkpoint_steps=PAPER_CHECKPOINT_STEPS, batch_size=BATCH_SIZE,
              grad_accum=1, learning_rate=DAPT_LR,
              warmup_ratio=WARMUP_STEPS / MAX_STEPS, lora=FINETUNE_LORA)
    pm = ft.finetune_dapt("physics", **kw)
    bm = ft.finetune_dapt("biology", **kw)
    return {
        "baseline": pm.loc[pm["index"] == 0, "checkpoint"].iloc[0],
        "physics-ft": pm.loc[pm["index"] == pm["index"].max(), "checkpoint"].iloc[0],
        "biology-ft": bm.loc[bm["index"] == bm["index"].max(), "checkpoint"].iloc[0],
    }


def pick_peak_layer(corr_by_model: dict[str, pd.DataFrame], feature: str = "pca") -> int:
    """Layer with the highest mean Spearman (across models) for ``feature``."""
    frames = [c[c["feature"] == feature][["layer", "spearman"]] for c in corr_by_model.values()]
    mean = pd.concat(frames).groupby("layer")["spearman"].mean()
    return int(mean.idxmax())


def run_aligned_comparison(av: pd.DataFrame, rm_cmp: pd.DataFrame) -> pd.DataFrame:
    """Specialized-attention source comparison (ΔLL) across all fixed-effects specs.

    For each spec, fit ``RT ~ <spec> + A_source + (1|reader)`` for the four
    attention sources (general baseline + residualized physics/biology/aligned) at
    the FT index, reporting ΔLL over the no-attention baseline. Reuses
    ``model_comparison_over_epochs`` (attention sits in its ``surprisal`` column);
    coefficient columns renamed to ``b/p_attention``.
    """
    cmps = []
    for spec in mc.MODEL_SPECS:
        cmp = mc.model_comparison_over_epochs(
            av, rm_cmp, None, measure=MEASURE, spec=spec,
            models=ALIGNED_MODELS, indices=[1],
        ).rename(columns={"b_surprisal": "b_attention", "p_surprisal": "p_attention"})
        cmp["spec"] = spec
        cmps.append(cmp)
    return pd.concat(cmps, ignore_index=True)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("Loading PoTeC word features + reading measures")
    words = data.load_word_features()
    rm_raw = data.load_reading_measures()           # carries is_expert + STTS_PoS_tag
    rm_cmp = rt.clean_reading_times(rm_raw, MEASURE)  # cleaned RT response (as main.py)
    et_all = co.build_et_table(rm_raw, participants="all")  # relative gaze + PCA

    # ── Step 1 — full fine-tuning (continued pre-training) ─────────────────────
    print("Step 1 — full fine-tuning physics + biology (GPU)")
    models_attn = finetune_full(NAME)
    print("  models:", {k: Path(v).name for k, v in models_attn.items()})

    # ── Step 2 — raw attention per model; per-layer gaze correlation ───────────
    attn_by_model, corr_by_model, corr_all = {}, {}, []
    for slug, ckpt in models_attn.items():
        print(f"\n=== {slug} ({Path(ckpt).name}) ===")
        model, tok = su.load_causal_lm(ckpt, attn=True)
        attn_df = at.extract_attention(words, model, tok, method="raw")
        attn_by_model[slug] = attn_df
        corr = co.correlate_attention(attn_df, et_all)
        corr.insert(0, "model", slug)
        corr_by_model[slug] = corr
        corr_all.append(corr)
        del model
        if su.torch.cuda.is_available():
            su.torch.cuda.empty_cache()

    corr_df = pd.concat(corr_all, ignore_index=True)
    corr_df.to_csv(PROJECT_ROOT / "results_attention_correlation.csv", index=False)

    # ── Step 3 — peak attention-vs-gaze layer ──────────────────────────────────
    L = pick_peak_layer(corr_by_model, "pca")
    print(f"\nPeak attention-vs-gaze layer (PCA, all readers): L={L}")

    def at_layer(slug: str) -> pd.DataFrame:
        a = attn_by_model[slug]
        return a[a["layer"] == L][WORD_KEY + ["attention"]]

    # ── Step 4 — robust domain-attention vs expert reading-time analysis ───────
    # Residualize each FT model's layer-L attention on the baseline → "specialized"
    # attention (the domain-specific component; surface signal removed), mirroring
    # Škrjanec et al.'s residualized specialized surprisal.
    print("Step 4 — robust domain-attention analysis")
    spec_pw = ap.build_specialized(
        at_layer("baseline"), at_layer("physics-ft"), at_layer("biology-ft")
    )
    av = ap.specialized_versions(spec_pw)  # general (idx0) + specialized phys/bio (idx1)

    # (a) Source comparison by ΔLL (specialized aligned vs single vs general baseline)
    cmp_all = run_aligned_comparison(av, rm_cmp)
    cmp_all.insert(0, "layer", L)
    cmp_all.to_csv(PROJECT_ROOT / "results_attention_aligned_comparison.csv", index=False)
    print("\n(a) source ΔLL (specialized attention):")
    print(cmp_all[["spec", "model", "delta_ll", "b_attention", "p_attention"]]
          .to_string(index=False))

    # (b) Reader-clustered Vuong significance (aligned vs each single source)
    vuong = pd.concat(
        [ap.aligned_attention_vuong(av, rm_cmp, spec=s) for s in ("full", "paper_full")],
        ignore_index=True,
    )
    vuong.to_csv(PROJECT_ROOT / "results_attention_vuong.csv", index=False)
    print("\n(b) reader-clustered Vuong (aligned vs single):")
    print(vuong[["spec", "comparison", "z", "p", "p_adj", "winner"]].to_string(index=False))

    # (c) PRIMARY: paper-09 three-way LRT — does aligned-specialized attention add
    #     over general attention, specifically for experts on technical terms?
    itest = ap.specialized_interaction_test(spec_pw, rm_cmp, measure=MEASURE)
    pd.DataFrame([itest]).to_csv(
        PROJECT_ROOT / "results_attention_interaction.csv", index=False)
    print("\n(c) specialized × expert × technical LRT:")
    print(f"    LRT χ²={itest['lr_stat']:.2f} (df={itest['df']}) p={itest['p_lrt']:.4g}"
          f"  |  3-way coef={itest['coef_3way']:.3g} p={itest['p_3way']:.4g}")

    # (d) Model-free robustness: ΔA (domain FT − baseline) vs Δ(expert speed-up)
    diff = ap.attention_expertise_diff_correlation(
        at_layer("baseline"), at_layer("physics-ft"), at_layer("biology-ft"), rm_raw,
        measure=MEASURE)
    diff.insert(0, "layer", L)
    diff.to_csv(PROJECT_ROOT / "results_attention_diff_correlation.csv", index=False)
    print("\n(d) ΔA vs Δ(expert speed-up) correlation:")
    print(diff.to_string(index=False))

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nFigures")
    fig, ax = plt.subplots()
    viz.across_models_attention_curve(corr_by_model, feature="pca", ax=ax)
    ax.set_title("raw attention vs gaze (PCA, all readers)")
    save_fig(ax, "attn_layer_curve_pca_all")

    for spec, g in cmp_all.groupby("spec"):
        fig, ax = plt.subplots()
        viz.across_models_bar(g, "delta_ll",
                              ylabel="ΔLL (RT ~ +specialized attention)", ax=ax)
        ax.set_title(f"specialized attention — {spec} (layer {L})")
        save_fig(ax, f"attn_aligned_comparison_{spec}")

    print(f"\nDone. Figures in {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
