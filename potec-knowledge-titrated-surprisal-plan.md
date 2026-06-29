# Knowledge-Titrated Personal Surprisal — Follow-up Experiment Plan

Follow-up to Škrjanec & Demberg (2026), *"Language models that match reader
experience are better predictors of reading times"* (JML 146, 104677).

---

## 1. Motivation / gap

The paper adapts a German LM (GerPT2) to a **group** — biology vs physics
students — and feeds each group "reader-aligned" surprisal. The `expertise`
predictor is **binary** (expert / novice). Two facts make a sharper, untried
experiment possible:

1. **PoTeC has per-participant scores.** Each reader answered text-independent
   background-knowledge questions *and* comprehension questions for every text.
   The paper collapses these into one binary `expertise` flag and discards the
   gradient.
2. **Amount of adaptation is a tunable knob.** Study 2 saves a checkpoint ladder
   (training steps 4, 16, 64, 256, 1024, 4096, 16384) and shows the best step
   count differs by reader group and reading measure.

Nobody has joined them. "Optimal amount of adaptation" is always answered at the
**population** level (Oh & Schuler ~2B tokens; this paper's sweeps). Background
knowledge is only ever a binary split or a regression random effect — never used
to **drive the LM itself**.

## 2. Core idea

Make the **degree** of LM domain adaptation a continuous function of each
reader's measured background-knowledge score. Each participant gets surprisal
from an LM checkpoint adapted to *their* knowledge level, not a one-size group
model.

## 3. Central hypothesis (falsifiable)

> For a given reader, the number of domain-adaptation steps `k*` that maximizes
> ΔLL (fit to *that reader's* RTs) is monotonically correlated with their
> background-knowledge score.

- High-knowledge reader → fit peaks at a **more** domain-adapted checkpoint
  (more steps).
- Low-knowledge reader → fit peaks **earlier** / nearer the general model.

If true: per-reader knowledge-calibrated surprisal beats BOTH the binary
expert/novice model AND any single population-optimal step count.

### Secondary predictions
- Effect strongest on **late** measures (GP, TFT), matching the paper's finding
  that reader-alignment helps late, not early.
- **FPRT (first-pass) stays flat** across `k` — built-in placebo channel. If the
  effect shows up in FPRT too, suspect a confound.

## 4. Why non-trivial (not "just add a covariate")

- Group-level result can be driven by a few strong experts. Per-reader
  calibration predicts a **within-group ordering** — strictly harder test.
- Turns "amount of adaptation" from a nuisance hyperparameter into a **cognitive
  measurement**: adaptation steps become a proxy for individual knowledge state,
  validated against an independent behavioral score.

## 5. Data + assets (all existing — minimal new compute)

| Asset | Source |
|---|---|
| PoTeC eye-tracking (75 participants, 6 texts) | osf.io/78jdq ; PoTeC paper (Behav Res Methods, doi 10.3758/s13428-024-02536-8) |
| Per-participant background-knowledge scores | PoTeC release |
| Per-participant comprehension scores | PoTeC release |
| GerPT2 base + biology/physics checkpoint ladders (FFT + adapters), 3 seeds | paper's release (osf.io/78jdq) |
| Word-level term annotations (level 0/1/2) | PoTeC |

No new LM training required if checkpoints are released. If not, reproduce ladder
(§9 fallback).

## 6. Method — step by step

### 6.1 Surprisal extraction
1. For each domain LM `D ∈ {bio, phys}`, each checkpoint step
   `k ∈ {4,16,64,256,1024,4096,16384}`, each of 3 seeds:
   - Tokenize PoTeC texts with GerPT2 tokenizer.
   - Subword surprisal `-log2 p(w_i | w_<i)`; sum subwords → word-level
     (chain rule), exactly as paper §"Surprisal estimates".
   - Average across the 3 seeds.
2. Result: surprisal table indexed by `(text, word, domain, step)`.

### 6.2 Per-reader knowledge score
- For each participant `p` and text `t`: extract PoTeC background-knowledge score
  `bg(p,t)` (text-independent) and comprehension `comp(p,t)`.
- Primary calibrator = `bg(p,t)`. Per-text, not per-discipline → avoids
  circularity with the discipline split.
- Scale + center within the analysis.

### 6.3 Find each reader's k*
For reading measure `M ∈ {FPRT, GP, TFT}`, reader `p`, domain LM matched to text
domain:
1. For each `k`, fit baseline + surprisal(k) regression and record
   `ΔLL_p(M,k) = LL_exp − LL_base`.
   - Use the paper's baseline (Eq. 2/4): `LogRT ~ Length + LogFreq + Position +
     Expertise*Terminology + (Surprisal) + random effects`.
   - To get a clean per-reader ΔLL, fit **per-participant** simple linear models
     (paper Appendix C/D shows LM ≈ LMER for these data — use LM for tractability
     and to avoid memory issues they flagged).
2. `k*_p(M) = argmax_k ΔLL_p(M,k)`.

### 6.4 Key tests
1. **Monotonicity (main):** Spearman correlation `ρ( k*_p , bg_p )` across
   readers, computed **within each discipline group** (controls discipline
   confound). Aggregate `bg_p = mean_t bg(p,t)`.
2. **Continuous head-to-head:** build a single per-reader "titrated" surprisal by
   selecting checkpoint `k(bg_p)` via a monotone map fit on a held-out split
   (nested CV across participants, §7). Compare three models:
   - (a) binary reader-aligned (paper's best),
   - (b) population-optimal single `k`,
   - (c) knowledge-titrated per-reader `k(bg_p)`.
   Compare with **Vuong non-nested test** (paper Appendix C, R `nonnest2`).
3. **Placebo:** repeat (1) on FPRT — expect null.

## 7. Cross-validation / leakage control

- **Participant-level nested CV.** The monotone map `bg → k` is fit on training
  participants only; ΔLL gains evaluated on held-out participants. Prevents
  fitting `k*` and testing on the same reader.
- Map family: start simple — bin `bg` into terciles → median `k*` per tercile;
  then monotone isotonic regression as richer variant. Pre-register both.

## 8. Controls, confounds, kill-criteria

| Risk | Mitigation |
|---|---|
| Discipline drives effect, not graded knowledge | Correlate **within** group only |
| Circularity with `expertise` predictor | Use per-text `bg`, keep `expertise` in baseline so titration must add *beyond* it |
| `k*` unstable / noisy per reader | 3-seed averaging; bootstrap CI on ρ; report per-reader ΔLL curves |
| Reading speed ∝ knowledge inflates fit | Length/Freq/Position already covaried; also report on length-residualized RT |
| Few participants per group (~32 phys / ~43 bio) | Report effect size + bootstrap, not just p; treat as confirmatory-power-limited |

**Kill-criterion (clean negative):** if `k*` is flat across readers (no
ρ, titrated model ≈ binary), graded knowledge does NOT map to adaptation amount.
Still publishable — bounds the binary model and the "amount of adaptation as
cognition" claim.

## 9. Fallback if checkpoints unavailable

Reproduce the ladder from the paper's recipe:
- Base: GerPT2-small (163M) — huggingface `benjamin/gerpt2`.
- Domain corpora: German Wikipedia + Spektrum.de scraped for biology & physics
  (paper §"Training data"). If scrape infeasible, substitute any matched-domain
  German corpus and re-anchor the step ladder by perplexity, not absolute steps.
- Full fine-tune (`transformers` causal LM) + bottleneck adapters
  (`adapter-hub/adapters`, reduction factor 16 → size 48), 3 seeds, checkpoint at
  steps {4,16,64,256,1024,4096,16384}, ≤16384 steps, batch 8, lr 1e-4, 100 warmup.

## 10. Deliverables

1. Surprisal tables per `(domain, step, seed)`.
2. Per-reader ΔLL curves over `k` for each measure (the headline figure:
   x = steps, y = ΔLL, one line per reader, colored by `bg` score).
3. Within-group ρ(`k*`, `bg`) with bootstrap CI, per measure.
4. Vuong test table: titrated vs binary vs population-optimal.
5. FPRT placebo result.

## 11. Expected result / payoff

- **If confirmed:** adaptation-step count is a readout of individual knowledge
  state; personalized surprisal beats group surprisal on late measures; "amount
  of adaptation" reinterpreted as cognitive measurement.
- **If null:** binary expertise is sufficient resolution; graded knowledge does
  not titrate processing-effort expectations linearly. Bounds the framework.

## 12. Novelty check (web, June 2026)

- JML 2026 paper + conference precursor: group/binary expert-novice only.
  https://www.sciencedirect.com/science/article/pii/S0749596X25000701 ;
  https://sfb1102.uni-saarland.de/publication/expert-adapted-language-models-improve-the-fit-to-reading-times/
- PoTeC documents per-participant background-knowledge + comprehension scores;
  unused for LM titration.
  https://link.springer.com/article/10.3758/s13428-024-02536-8
- "Optimal amount of training/adaptation" = population-level only (Oh & Schuler
  ~2B tokens). https://arxiv.org/pdf/2304.11389 ; https://arxiv.org/pdf/2402.02255
- Closest neighbor "Reverse-Engineering the Reader" tunes LM to fit reading data
  but aggregate psychometric alignment, NOT per-individual adaptation gated by an
  external knowledge score. https://arxiv.org/pdf/2410.13086

No hit ties **individual background-knowledge magnitude** to **per-reader degree
of LM adaptation**. Idea untried.
