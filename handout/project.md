# title

## abstract

## introduction

- exposition
- relevant prior work:
  - surprisal theory
  - overview on which models work best (smaller, pythia paper)
  - effects of fine-tuning for a task
  - zero-shot (prompts?) learning

- research question, why it is novel
- main findings

## data

### potec dataset

- overview of the dataset
  - the task
  - number of readers
  - number of datapoints (words)

- data cleaning methods
  - IQR cleaning (slow reading times)
  - dropping 0 reading-times
  - start-of-text(sentence?), end-of-text(sentence?) cleaning

### German-commons dataset

- overview of the dataset
- pre-filtering methodology
- fine-grained filtering methodology

### model - GPT-2

- base weights
- number of parameters
- initial perplexity on the three test-sets (biology, physics, general)

### model - Llama

- base weights
- number of parameters
- initial perplexity on the three test-sets (biology, physics, general)

## experiments and results

### fine-tuning methodology

- number of parameters tuned
- perplexity curve for each domain on a fixed test-set

### zero-shot (prompting)

- example: the probabilities (color coded) on a sentence with / without a prompt to influence them.

### linear regression fit

- linear regression model(s)
- justification for all other features (word freq, word length, etc.)
- statistical tests used

### results

## discussion and conclusion

### key results

### theoretical implications

### limitations of the work

### future work

## bibliography

## appendix

- fine-tuning hyperparameters (GPT-2, Llama)
- german-commons fitering hyperparameters
