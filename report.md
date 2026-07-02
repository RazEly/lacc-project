Read paper (all 21 pages), all src/ modules, results_slim.csv. Review below.

Critical

1. LogFreq computed but never in model. model_comparison.py:54 builds log_word_freq;_BASE_TERMS (model_comparison.py:93) = word_length + word_position + is_expert * is_technical — no frequency term. Paper Eq. (2) has LogFreq in every model. Same for log_word_length (:55, also unused). Consequence: surprisal absorbs frequency variance → all surprisal ΔLLs inflated. Worse for your DAPT story: paper argues early adaptation mostly re-learns domain word frequencies — so D_aligned may be a frequency effect that a freq covariate would kill. Biggest single fix: add log_word_freq to_BASE_TERMS.

2. Llama null claim needs more than nonsignificance. results_slim.csv: llammlein aligned ΔLL negative everywhere, p_lrt=1. That shows "no detectable gain," not "effect does not hold." Two problems:

- "Significant for GPT-2, not for Llama" is not itself a test (Gelman & Stern). Need direct comparison: fit both models' D terms jointly, or bootstrap/CI the ΔLL difference, or equivalence test (TOST) / Bayes factor on b_resid.
- Comparison currently confounded (see 3). Fix confounds before claiming size effect.

1. GPT-2 vs Llama adaptation not matched.

- finetune.py:294-309: german-gpt2 gets vocab resize → modules_to_save=["wte"] → full embedding matrix trains; LLäMmlein (no resize) trains LoRA only, embeddings frozen. Paper says final layers/embeddings carry frequency learning — GPT-2 gets exactly the capacity that matters, Llama doesn't. Asymmetric by construction.
- tokens_per_step: gpt2 = 512·8 = 4096; llama = 512·2·3 = 3072. Same checkpoint index ≠ same tokens seen (step 4096: 16.8M vs 12.6M).
- Same LR (2e-4) for 124M and 1B; 1B typically wants smaller. Untuned LR difference alone can produce your null.
- No adaptation diagnostic reported. Paper's Figs 4–5 (perplexity + technical-term surprisal trajectories) establish adaptation happened before interpreting RT fits. Manifest has perplexity — plot it per model. If LLäMmlein's technical-term surprisal barely moves, the null is about your DAPT, not about big models.

1. D-term asymmetry undertests the hypothesis of interest. S_baseline gets full 3-way interaction ({col} *is_expert* is_technical, :94); residual D_<name> enters as plain slope (:119). Paper's core finding lives in surprisal × expertise: reader-aligned models predict who slows down where. If adaptation's gain is expert/novice-differential (it should be), a main-effect-only D misses it. Give D the same interaction structure (3 extra df, still nested LRT) — or at least D * is_expert.

Major

1. TFT only. Paper's headline is the early/late dissociation: FPRT ~flat, GP/TFT benefit. Your "does not hold for Llama" claim is much stronger shown across FPRT/GP/TFT (does Llama fail only where GPT-2 succeeds?). RPD_inc (=GP) and FPRT already loaded in data.py. Note clean_reading_times fences on one measure — re-clean per measure like paper does.

2. Single seed. Paper averages surprisal over 3 seeds per method×domain×step because "language model training is sensitive to the initial state of weights." You run seed=0 once. LoRA is noisy at small step counts; your gpt2 aligned peak (ΔLL 3.1 at step 64) vs llama trough could partly be seed luck.

3. Multiple comparisons over checkpoint sweep. gpt2 aligned p_lrt: .055, .024, .013, .036, .042, 1.0 across 6 checkpoints. None survives Bonferroni (α=.0083). Paper's protocol: pick best checkpoint by ΔLL first, run one test — explicitly "avoiding the problem of multiple comparisons." Either adopt that, or correct, or treat the sweep as descriptive (paper frames ΔLL as "numeric tendency, not hypothesis testing").

4. No aligned-vs-alternative direct test. Paper's Study 3 uses Vuong (nonnest2) for non-nested pairs (reader- vs text-aligned). You compare each source only against base surprisal; "aligned beats prompted" is currently eyeballed from ΔLL (1.8–3.1 vs 1.5 — barely different). Your residual trick allows a cleaner move: fit base + D_aligned + D_prompted jointly, LRT each term given the other. Also: you dropped text-aligned entirely — it's the paper's main contrast and cheap for you (columns already cached: s_phys/s_bio by text domain).

5. Training horizon short for adapter-style methods. Max 4096 steps (main.py:26); paper's TFT optimum for adapters was 4096 with signs of gains past 16384. LoRA ≈ adapter regime → you may be truncating pre-peak, again especially for the 1B model (bigger models adapt slower per step). Extend schedule to 16384 (one more 4ⁿ point) before concluding.

Minor

- Continuous predictors not scaled/centered (paper scales all; only word_position standardized). Doesn't change LRT but helps convergence + makes β comparable to paper's Table C tables.
- results_slim.csv is stale: lacks b_resid/p_resid columns the current_row emits. Regenerate before reading numbers off it.
- surprisal.py:88-89: prompt path concatenates tokenizer(prompt).input_ids + tokenizer(words).input_ids. If LLäMmlein's tokenizer adds BOS, you get stray mid-sequence BOS. Check tokenizer.add_bos_token; strip specials on one side.
- Prompt experiment thin: one 6-word persona prompt, base (non-instruction-tuned) GPT-2 — persona framing is near-meaningless to a raw LM. Prompt effects are notoriously variance-heavy: use several paraphrases + longer domain-content prompts (e.g. a domain paragraph as context), treat prompt as random factor or report range. Also try prompt × DAPT combined.
- fp16 logits → log_softmax float (surprisal.py:97): fine, but if any result is marginal, spot-check fp32.
- Same checkpoint index for physics and biology LMs (paper sweeps full 7×7 grid). Acknowledged simplification — just say so; the paper's grid shows off-diagonal optima exist.

Suggested order of work

1. Add log_word_freq (+ scaling) to _BASE_TERMS; rerun. Everything downstream may change.
2. Add D * is_expert (or full 3-way) interaction; rerun.
3. Add FPRT + GP.
4. Fix Llama parity: no-resize embedding handling symmetric (either train embeddings both or neither), match tokens/step, small LR sweep for 1B; plot perplexity + technical-term surprisal trajectories as adaptation evidence.
5. 3 seeds, average surprisal.
6. Then the claim tests: best-checkpoint selection → single LRT per measure; direct GPT-2-vs-Llama ΔLL comparison with CI (bootstrap over readers) or TOST on b_resid.
Strong points worth keeping: residual/split-signal design is statistically cleaner than paper's non-nested ΔLL eyeballing; row-set consistency across sources (shared dropna) makes ΔLLs genuinely comparable; ML fits for LRT correct; deviation coding correct; cleaning matches paper.
