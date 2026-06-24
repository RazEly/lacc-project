# BERT surprisal & attention vs. PoTeC reading times — analysis summary

Notebook: `bert_surprisal_vs_reading_time.ipynb`
Data: PoTeC (Potsdam Textbook Corpus) — 12 German textbook texts (6 biology `b0–b5`,
6 physics `p0–p5`), eye-tracking from biology/physics students.
Model: `bert-base-german-cased` (loaded with `attn_implementation="eager"` so
attentions are available). All reading-time work uses mean **TFT** unless noted.

---

## 1. Fresh BERT pseudo-surprisal (§1–4)
Recomputed word-level surprisal from scratch (the shipped `*_surprisal_bert-base`
column has a known bug). BERT is bidirectional → **pseudo-surprisal**: mask each
sub-token, read `-log2 p(token | rest of sentence)`, sum sub-words to the word.
Context = single sentence.

- Pearson r = **0.299**, Spearman ρ = **0.388** (n=1895, p≈2e-40). Positive, weak,
  significant — surprising words are fixated longer.

## 2. Regression + binning (§5, à la Pset 2)
OLS `mean_TFT ~ bert_surprisal`: slope **8.62 ms/bit**, t=13.6, **R²=0.089**.
Surprisal alone explains ~9% of RT variance (expected; length/freq/skips dominate).
Scatter shown without binning (jointplot) and with binning (`regplot x_bins=15`).

## 3. Expert vs. non-expert readers (§5.1)
PoTeC `expert_reading_label_numeric`: 1 = reader's discipline == text domain.
Surprisal identical per word; only human RT differs.

| group | slope (ms/bit) | R² | Spearman ρ | mean RT |
|---|---|---|---|---|
| all | 8.62 | 0.089 | 0.388 | 633 ms |
| expert | 6.90 | 0.075 | 0.360 | 573 ms |
| non-expert | 9.63 | 0.091 | 0.389 | 666 ms |

Experts read ~90 ms faster and show a **weaker** surprisal effect → domain knowledge
buffers the predictability cost.

## 4. Domain-specific vocabulary (§5.2)
Domain word = `is_expert_technical_term` OR `is_general_technical_term`
(404 / 1895 words; 155 strict expert jargon). 2×3 table (vocab × reader group):

- Domain words much slower (mean RT 1149 vs 493 ms).
- **Expertise blunts surprisal only on domain words**: domain slope expert 5.36 vs
  non-expert 9.70; on non-domain words slopes are near-identical (4.15 vs 4.63).
- General-LM surprisal tracks the **novice** better on jargon (ρ 0.249 vs 0.187).

The expert×domain interaction is the clean result: knowledge cancels the surprisal
cost precisely on technical vocabulary, not on shared general words.

## 5. Raw attention by layer (§6, replicating Mouratidi & Poesio 2025, Fig. 1)
Raw attention, head-averaged, attention each word *receives* (mean over queries),
sub-token→word by max (Sood et al. 2020), normalized per sentence; Spearman vs RT
per layer.

- **Layer 0 (first): ρ=0.249** — strongest early layer (matches paper).
- Middle layers collapse (~0, L3 n.s.).
- **Layer 11 (last): ρ=0.288** — global max here.

Partial replication: early-layer bump reproduced; global peak sits at the last layer
(paper's BERT peaked at layer 1). Likely because TFT is a late/cumulative measure —
swapping to FFD/FPRT should favor early layers more.

## 6. Domain fine-tuning test — DID IT REVERSE THE EXPERT DISADVANTAGE? (§7) — **INCONCLUSIVE**
Hypothesis tested: *"surprisal of a BERT fine-tuned on the domain correlates with a
domain expert's reading more strongly than a novice's."*

Design: **leave-one-text-out (LOTO)** — for each text, MLM-fine-tune a fresh model on
the other 11 texts, recompute surprisal on the held-out text (to avoid scoring
sentences the model memorised).

Findings (with word length + log-frequency controls):
- Adaptation appeared to work (surprisal dropped more on domain words, −5.4 vs −3.5 bits).
- But raw ρ on domain words barely moved (expert 0.185, non-expert 0.258).
- Controlled: general surprisal predicts **neither** group's domain reading (the §5.2
  raw domain correlation was largely a length/frequency confound); domain-FT surprisal
  is significant **only for the non-expert** (p=0.028), never the expert.

### Why it's inconclusive (§7.1 / §7.2 — coverage diagnostic)
**~88% of domain words — and 94% of expert technical terms — appear ZERO times in
their LOTO training set** (median training occurrences = 0). Domain jargon is
text-specific, so removing the held-out text strips out exactly the words being scored.
**The model was never exposed to the jargon it was tested on → more epochs cannot help;
the problem is missing data, not undertraining.** So §7 does NOT refute the hypothesis —
it leaves it untested.

---

## Key takeaways
1. Surprisal → RT: weak but robust (R²≈0.09); experts less sensitive than novices.
2. The expertise effect is **specific to domain vocabulary** (clean interaction in §5.2).
3. BERT attention aligns with eye-gaze most in the **first layer** (+ a last-layer rise);
   replicates the encoder finding.
4. The domain-fine-tuning hypothesis remains **OPEN** — our LOTO test was fatally
   under-covered for the jargon that matters.

## NEXT STEP — required to actually test the fine-tuning hypothesis
**Integrate a LARGE data source UNRELATED to PoTeC and train on it — to prevent leakage.**

- Continue-pretrain (MLM) `bert-base-german-cased` (or `deepset/gbert-base`) on a large
  **external** German biology + physics corpus — e.g. German Wikipedia Biologie/Physik
  category dumps, German scientific abstracts, OpenLegalData-style domain text, or a
  domain slice of OSCAR/mC4. It must be **disjoint from the PoTeC stimuli**.
- This gives two things LOTO could not: (a) **real exposure to the technical
  vocabulary** (the jargon appears many times in a large corpus), and (b) **no
  sentence-level leakage** (the model never sees the exact PoTeC sentences it is scored
  on, so pseudo-surprisal does not collapse to ~0).
- Then recompute PoTeC surprisal with the adapted model and rerun §5.2 / §7 (the 2×3
  table + length/frequency-controlled comparison).
- Also check a possible ceiling: if experts read jargon at floor speed (low RT variance),
  range restriction limits any achievable correlation regardless of model quality —
  report expert RT variance on domain words alongside.

## Files
- `bert_surprisal_vs_reading_time.ipynb` — full analysis (executed; mise314 kernel).
- `finetune_domain_surprisal.py` — LOTO fine-tuning script (writes the TSV below).
- `bert_domain_surprisal_loto.tsv` — per-word domain-fine-tuned surprisal (1895 words).
- `loto.log` — fine-tuning run log.
- `ANALYSIS_SUMMARY.md` — this file.
