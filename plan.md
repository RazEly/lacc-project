## step 1 - reading time

Setup:

- words dataframe: word, word length, location in text, index in text, text id, text domain, whether its a technical term or not
- readers dataframe: same as words, include: reader id, reader reading time, information about the readers domain (what level, what field, binary: expert for that field yes/no)
- merge the words and readers dataframe together if not already present in the dataset.

Cleaning:

- drop the first and last words of each sentence, across all readers.
- drop skipped words (according to the dataset - reading time = 0)
- apply the filtering in Smith & Levy 2013 (IQR outliers)

Aggregation:

- group by to get mean, standard deviation, median reading time for texts in each domain:
  - everyone (baseline)
  - only domain experts (physicists if the text is physics, biologists if the text is biology)
  - each of the different levels of the domain experts (undergrad, graduate, phd etc. - according to the reader metadata saved)

## step 2 - model surprisal

Setup:

- decoder-only causal LMs only (e.g. Llama). use transformers library. make it model-agnostic within decoder family — able to plug in any decoder model:
  - default huggingface weights
  - custom weights that will be extracted later during fine-tuning.
- use the last layer of the model to get the probability for each word in each sentence
  - options: prompt=None|str: if string, accept this as a system prompt before the calculation of surprisal. domain-matched: physics prompt for physics texts, biology prompt for biology texts.
cleaning:

- drop the first and last words of each sentence.

## step 3 - attention

Extract attention representations per word per layer:

- **raw attention**: tokenize + forward pass (no system prompt). average attention scores across heads per layer. for subtoken words: assign each word the max attention score among its subtokens (Sood et al., 2020). final score = average attention the word receives from all other words in the sentence, normalized by sum (relative attention per sentence).
- **attention flow**: treat raw attention graph as flow network (Edmonds-Karp max-flow algorithm). last token = target "sink". use reduced number of paths to respect causal attention structure. apply position-based decay weighting (Metzger et al., 2022) to correct early-token bias. normalize at word level by combining subtokens.

Eye-tracking features used for comparison (word-level, normalized within sentence, averaged across participants):

- GD (gaze duration), TRT (total reading time), FFD (first fixation duration), SFD (single first duration), GPT (go-past time), F (fixation count)
- PCA reduction: fit PCA on the 4 most informative features (drop SFD and GPT — sparse/noisy). fit separately per domain (physics texts / biology texts), following paper methodology (Mouratidi & Poesio, 2025). retain 1 component (~94-97% variance explained). each word represented by its score on this component.

Model attention extraction:

- extract without masking (standard forward pass, no system prompt)

## step 4 - fine tuning

- accept a baseline German decoder-only causal LM from transformers library
- fine-tuning objective: causal next-token prediction (continued pre-training / DAPT — Gururangan et al., 2020)
- fine-tuning data: `german-commons` corpus, split by domain (physics / biology). separate from PoTec texts used in reading time analysis (no leakage).
- finetune the baseline model on physics subset, then on biology subset independently.
  - save the finetuned model after every 10 epochs
  - save the number of words processed in those 10 epochs: how many words the model has been fine-tuned on.
  - save the dataframe from step 2 with different versions of the model
  - ADD: ways to show how far along the fine-tuning is. for example: how perplexity goes down as the epochs progress.

## step 5 - analysis

Helper:

- function that accepts a dataframe of surprisals (step 2 result) and a dataframe of reading time.
- merges them (inner join).

Correlation (surprisal):

- function that accepts a merged dataframe (from helper) returns pearson, spearman correlation
- options:
  - domain='physics'/'biology'/'all' - which texts to account.
  - domain_only=True/False - if true, filter to only explain domain words, not all words. If false: explain all words (no filtering)
  - mode='mean'/'median' mode.
  - participants= 'experts'/'novices'/'all' - filters to only that group of people, according to the text domain.

Correlation (attention):

Methodology: for each attention method (raw, attention flow), correlate attention scores with eye-tracking measures word-by-word using Spearman's r.

- function that accepts: attention scores dataframe (step 3), eye-tracking dataframe (step 1), layer index or 'all'
- for each layer: compute Spearman r between attention scores and each eye-tracking feature individually, and with the PCA component
- return: layer-wise correlation table (layer × feature × attention_method)
- options:
  - `attention_method='raw'/'flow'` — which representation
  - `layer=int/'all'` — single layer or all layers
  - `domain='physics'/'biology'/'all'`
  - `participants='experts'/'novices'/'all'`

Regression:

- function that accepts merged dataframe.
- fit a linear regression from `statsmodels` explaining reading time
- options:
  - mode='mean'/'median' mode.
  - participants= 'experts'/'novices'/'all' - filters to only that group of people, according to the text domain.
  - domain='physics'/'biology'/'all' - which texts to account.

TODO: checking statistical significance of the correlations and regression coefficients, using appropriate tests (e.g., t-tests, ANOVA) and correcting for multiple comparisons if necessary.

## step 6 - visualization
