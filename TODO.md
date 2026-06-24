# TODO

## Per-person regression (not in plan yet — design undecided)

Two hypotheses to test when this section is added back:

- **A** (model comparison): domain-expert model surprisal predicts expert reading times better than baseline surprisal → compare R² / AIC between models. Claim: domain adaptation captures expert lexical expectations.
- **B** (interaction): surprisal-RT correlation increases for experts but not novices as fine-tuning progresses → interaction term: fine-tuning stage × reader expertise. Claim: fine-tuning makes model more cognitively plausible specifically for domain readers.

Model structure also undecided: random intercepts only vs. random slopes for surprisal by reader (see Barr et al., 2013).

---

## Papers to read / cite

- **Gururangan et al. (2020)** — "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks". Justification for DAPT (continued pre-training on domain text) as fine-tuning objective in Step 4.

- DAPT
- LORA
-
