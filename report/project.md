# title

## abstract

## introduction

### surprisal theory

Surprisal theory () establishes the connection between human reading time to surprisal and remains consistent across different language models.
Previous works () explore the relationship by using statistical language models (n-grams), and more recent works have explored it using modern-day large language models ().

### richer models - poorer fit

It has also been established () that while the linear relationship holds, richer parameter models provide a poorer fit to human reading times.

### domain-expertise

In this project, we continue exploring the surprisal theory, in the context of reading differences between domain-experts and novices.
[It is safe to assume] that domain experts would read texts taken from their own field of expertise faster than novices, who have not been exposed to said domain as throughly, if at all.

We will be using the Postdam Textbook Corpus (Jakobi et al., 2025), that []. The corpus contains reading data from physics and biology students, reading texts from both domains. We will be treating students reading texts in their own domain of study as "experts" and compare their gaze patterns with students of the other domain.

#### ARTICLE QUOTE

Processing difficulties due to limited lexical knowledge manifest in online measures like reading behavior. For instance, encountering individual unfamiliar or infrequent words leads to slower reading and increased revisits (Just & Carpenter, 1980; Lowell & Morris, 2014). This holds for familiarity with the vocabulary of specific domains as well. Scientific texts frequently incorporate specialized and less common vocabulary, requiring the understanding of domain-specific concepts and relations between them.

Jian and Ko (2014) report that readers with a higher level of background knowledge in physics have shorter first-pass reading times and lower regression rates and spent less time rereading in comparison to lower-knowledge readers. Additionally, encountering information that is inconsistent with the domain knowledge of the reader shows effects on neural and behavioral measures, eliciting a greater N400 amplitude (Troyer & Kutas, 2018; Troyer et al., 2020) or longer reading with more regressions (van Moort et al., 2020).

### fine-tuning

Modern large language models have their own form of "domain-expertise", which often come in the form of domain-specific fine-tuning (). By using any of the
[The cognitive motivation is clear -] fine-tuning techniques () often freeze earlier layers of LLMs, adapting only the later layers to the desired domain, similar to how we would image a human studying - an update of one's knowledge base without a complete overhaul of one's existing knowledge.
While we do not attempt to make the claim that the two processes are the same, [they provide motivation]

#### ARTICLE QUOTE

adapters offer a parameter-efficient method for transferring the learned features and learning the new domain. In our study, we use the so-called bottleneck adapters (Houlsby et al., 2019). In the bottleneck adapter approach, new layers (adapter weights, which consist of feed-forward layers) are inserted between existing layers of the pre-trained model. These weights are of smaller dimensions than the pre-trained model. During training, the layers of the pre-trained model are frozen and remain intact, while the weights of the adapters are updated with respect to the loss function on the training data. The role of the adapter weights is to provide transformations of frozen pre-trained LM given the target domain. For even larger pre-trained LM, full fine-tuning is associated with high computational costs, while adapters achieve a similar performance (Pfeiffer et al., 2020) with fewer resources.

### zero shot learning

In addition to fine-tuning techniques, modern LLMs leverage [zero/few-shot-learning] (). It has been shown that a prior prompt to an LLM can greatly effect the generated outcome.

### previous work + our study

A recent study has analyzed the relation between fine-tuned language models and domain-expertise (). In this paper, the researchers explore the effect of aligned-surprisal on a linear regression fit, with respect to both text-domain and reader-domain alignment. They have found that reader-aligned surprisal provides a better fit to reading times by examining the GPT-2 () LM
The mentioned paper ()
In this paper, we will be re-producing the main findings of (), and expanding them by asking the following questions:

- Does the predictive power of reader-aligned surprisal remain consistent under larger language models?
- Does the predictive power of reader-aligned surprisal remain consistent under [zero/few-shot-learning], with no direct modification of the weights?

### main findings

## data

### PoTeC dataset

#### overview

PoTeC is an eye-tracking-while-reading dataset using stimulus materials adapted from German university-level textbooks on either physics or biology.

It consists of 12 texts, 6 from each domain. It involves eye-tracking data from 75 biology (32) and physics (43) students, all of which are all native German speakers.

The dataset includes the results of a reading comprehension multiple-choice exam, taken by each of the participants.
For the purpose of our study we will not be using the results of the exams, and consider a reader as an expert if the stimuli text domain aligns with their field of study.

In total, the dataset contains 142,125 data points (75 participants [X] (954 + 941) words). We used the PoTeC existing preprocessing scripts ([gh]) to extract reading measures. The measure we will examine in the scope of this study is total fixation time (TFT)

In addition, each stimuli text is labeled in one of three levels: common words, generally known, and domain-specific words. [distribution of the word types??]

#### data cleaning methods

For each reader, the first and last words in every sentence are dropped from the analysis. In addition, words with TFT=0 were dropped as well.
We have aggregated the dataset per-reader, calculating each participants mean total-fixation-time, and removed data points of TFT over 3 times the standard deviation of said participants

#### data points after cleaning

This results in a final dataset of
[TABLE: potec post cleaning]

### Fine-tuning dataset

We used `wikipediaapi` to scrape two domain-specific datasets.
For the initial search, we used expert-labeled terms from the PoTeC dataset. To enrich the initial list even further we used additional hand-picked seeds, consisting of course names from a standard B.Sc University curriculum from each domain (for example: [ADD]).
Upon visiting some term's Wikipedia page, our scraper collects all links within that article and uses them to further enrich the article pool.
We set the search depth to 2, to avoid domain drift and simplify future data cleaning tasks.

[TABLE: dataset statistics]

### model - GPT-2

The base model used is GPT-2 () with [approx] 124M parameters. We initialized the weights with `dbmdz/german-gpt2` from huggingface ().

- initial perplexity on the three test-sets (biology, physics, general)

### model - Llama

We used a [LLama 3.2 1B] model. We initialized the weights with `LSX-UniWue/LLaMmlen_1B` ().

- initial perplexity on the three test-sets (biology, physics, general)

## experiments and results

### fine-tuning methodology

#### LoRA fine-tuning

The fine-tuning methodology differs from the aforementioned study, instead of Houlsby adapters (), we used LoRA () adapters
[EXPLANATION ON LORA]

- perplexity curve for each domain on a fixed test-set

#### number of parameters used

#### note

The exact training hyperparameters are listed in the appendix.

### zero-shot (prompting)

Establishing a role-driven prompting methodology proved to be a challenging task;
While observing the change in surprisal for a role prompt such as "You are a Physics/Biology expert." might be a desirable research question by itself, the models were not training on instruction-following or chat data (), and therefore using such prompts would have been inappropriate in the context of this study.
Using instruct-models instead would provide a new challenge: since our domain-specific scraped datasets are free-text, the fine-tuning might [EXPLAIN WHY THIS IS A PROBLEM].

We settled on the following approach: using the same scraped datasets, we assigned random sentences as priors to the model, and observed the change in surprisal only for the subsequent sentence.
We aligned the domain of the prior with the domain of the reader, such that a physics reader would recieve a physics prior, regardless of the stimuli text.

We set the length of the prior to be a full sentence, capped at 128 tokens, approximately half of the length of the mean stimuli text.
To reduce variance, we repeated the process 20 times, and averaged the surprisal values for each word in the stimuli text.

The new surprisal values generated by the prompted model cannot be compared directly with the baseline model, since context lengths differs.
[!!!] Previous studies have found that longer context [affects] human reading-time fit negatively, with shorter context providing a better fit, we therefore expected the prompted models to provide a worse fit to the total fixation time.
[!!!] To combat this confounding factor, we introduced two pseudo-baselines, where we used a prior physics or biology prompt to all readers, regardless of their domain of study. This allows us to compare the effect of a domain-aligned prompt to a domain-misaligned prompt, while keeping the context length constant.

[example: the probabilities (color coded) on a sentence with / without a prompt to influence them]

### linear regression fit

Our regression model methodology follows the same approach as previous works ().

We used the `pymer4` package to fit a linear mixed-effects model to the selected reading measure, with the following features:

- word length
- word position in text
- word frequency: as estimated by dlexDB () [!!!]
- terminology: according to the PoTeC dataset
- expertise: an indicator for whether the readers domain of studies aligns with the domain of the stimuli text
- terminology * expertise: an interaction term
- surprisal: as estimated by the language model

the fitted model is of the form:
$$Log\,TFT \sim Length + LogFreq + Position + Expertise * Terminology + (1 | SubjectID) + (1 + Expertise | WordID) \tag{2}$$

[!!!] The features were then normalized using a z-score transformation, and the indicators were sum-coded: [!!!]

### methodology

We start by fitting a baseline model, with no surprisal term, and then fit a second model with the surprisal term included. We then compare the two models using a likelihood ratio test, to determine whether the surprisal term significantly improves the model fit.
We then introduce an aligned-surprisal term, where the surprisal is estimated by a model that has been fine-tuned on the same domain as the stimuli text, and compare it to the baseline surprisal model.
We re-fit the linear mixed-effects model with aligned-surprisal terms, produced by different checkoints of the fine-tuning process, taken every [4^n] up to [4,096].

Finally, to determine statistical significance, we compare the fit of every checkpoint to the baseline-surprisal model (surprisal term is present, base model weights) using a Vuong test, reporting the z-statistic and p-value.

### results

#### GPT results

#### Llama results

## discussion and conclusion

### key results

### theoretical implications

### limitations of the work

### future work

## bibliography

## appendix

- fine-tuning hyperparameters (GPT-2, Llama)
- german-commons fitering hyperparameters
