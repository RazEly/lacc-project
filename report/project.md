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

Modern LLMs have their own form of "domain-expertise", which often come in the form of domain-specific fine-tuning ().

[The cognitive motivation is clear -] fine-tuning techniques () often freeze earlier layers of LLMs, adapting only the later layers to the desired domain, similar to how we would image a human studying - an update of one's knowledge base without a complete overhaul of one's existing knowledge.
While we do not attempt to make the claim that the two processes are the same, [they provide motivation]

#### ARTICLE QUOTE

adapters offer a parameter-efficient method for transferring the learned features and learning the new domain. In our study, we use the so-called bottleneck adapters (Houlsby et al., 2019). During training, the layers of the pre-trained model are frozen and remain intact, while the weights of the adapters are updated with respect to the loss function on the training data. The role of the adapter weights is to provide transformations of frozen pre-trained LM given the target domain. For even larger pre-trained LM, full fine-tuning is associated with high computational costs, while adapters achieve a similar performance (Pfeiffer et al., 2020) with fewer resources.

### zero shot learning

An additional angle that we wish to explore in this study is the effects of a prior prompt on the predictive power of surprisal.
Crafted instructions, often referred to as "prompt engineering" is a standard practice in improving LLMs performance on downstream tasks, and was explored in numerous papers [large languge models are zero-shot reasoners, gpt-3 paper]

!!! In modern LLMs those instructions often come in the form of persona-instructions (), yet those techniques further back. In () providing numerous examples prior to a completion task improves LLM performance as well.

As far as we're aware no attempt has been made to leverage this quality to examine the change in the predictive power of surprisal on reading time as caused by a prior prompt.

### previous work + our study

A recent study has analyzed the relation between fine-tuned language models and domain-expertise ().
In this paper, the researchers explore the effect of aligned-surprisal on a linear mixed-effects regression model fit, with respect to both text-domain-alignment (the surprisal values are taken from two models, chosen according to the text domain that is being read) and reader-domain alignment (the surprisal values are taken from two models, aligned with the readers' domain of expertise).
The researchers have found that reader-aligned surprisal provides a better fit to reading times predictions.

In this paper, we will be re-producing the main findings of (), and expanding them by asking the following questions:

- Does the predictive power of reader-aligned surprisal remain consistent under larger language models?
- What is the effect of reader-aligned zero-shot learning techniques on the predictive power of surprisal?

We approach those questions by following the papers methodology, reproducing the main results, and replicating them on a larger language model (Llama 1B) ().
We then include prior prompts, aligned by readers' domain of expertise and examining their effects on surprisal values and the linear model's fit.

We have included a detailed comparison in the appendix comparing our methodology to aforementioned paper. [!!!]

### main findings

## data

### PoTeC dataset

#### overview

PoTeC is an eye-tracking-while-reading dataset using stimulus materials adapted from German university-level textbooks on either physics or biology.

It consists of 12 texts, 6 from each domain. It involves eye-tracking data from 75 biology (32) and physics (43) students, all of which are all native German speakers.

The dataset includes the results of a reading comprehension multiple-choice exam, taken by each of the participants.
For the purpose of our study we will not be using the results of the exams, and consider a reader as an expert if the stimuli text domain aligns with their field of study.

In total, the dataset contains 142,125 data points (75 participants [X] (954 + 941) words). We used the PoTeC existing preprocessing scripts ([gh]) to extract reading measures. The measure we will examine in the scope of this study is total fixation time (TFT)

In addition, each stimuli text is labeled in one of three levels: common words, generally known, and domain-specific words (labeled levels 0, 1 and 2).

#### data cleaning methods

For each reader, the first and last words in every sentence are dropped from the analysis. In addition, words with TFT=0 were dropped as well.
We have aggregated the dataset per-reader, calculating each participants mean total-fixation-time, and removed data points of TFT over 3 times the standard deviation of said participants

#### data points after cleaning

This results in a final dataset of
[TABLE: potec post cleaning]

### Fine-tuning Corpora

We used `wikipediaapi` to scrape two domain-specific datasets, and one neutral dataset.
For the initial search, we used expert-labeled terms from the PoTeC dataset. To enrich the initial list even further we used additional hand-picked seeds, consisting of course names from a standard B.Sc University curriculum from each domain (for example: [ADD]).
Upon visiting some term's Wikipedia page, our scraper collects all links within that article and uses them to further enrich the article pool.
We set the search depth to 2, to avoid domain drift and simplify future data cleaning tasks.
The initial scraping results consisted of approximately 6,000 articles from both domains. We then observed the pool of documents, deleted duplicates, and removed articles outside of the desired domain. In addition, math symbols and references were stripped from the text.

[TABLE: dataset statistics]

### model - GPT-2

The base model we used is `GerPT-2`, a German-only GPT-2 based model. (`benjamin/gerpt2`, huggingface)

- initial perplexity on the three test-sets (biology, physics, general)

### model - Llama

For the larger model, We used `LLaMmlen-1B` (), a German-only Llama based model.

- initial perplexity on the three test-sets (biology, physics, general)

## experiments and results

### fine-tuning methodology

#### LoRA fine-tuning

- perplexity curve for each domain on a fixed test-set

#### number of parameters used

#### note

The exact training hyperparameters are listed in the appendix.

### zero-shot (prompting)

Establishing a role-driven prompting methodology proved to be a challenging task;
While observing the change in surprisal for a role prompt such as "You are a Physics/Biology expert." might be a desirable research question by itself, the models we used were not trained on instruction-following or chat data (), and therefore using such prompts would have been inappropriate in the context of this study.
Using instruct-models instead would provide a new challenge: since our domain-specific scraped datasets is unstructured text, extracted from wikipedia, training on such corpora might [result in catastrophic forgetting, causing the adapted models to "forget" their instruction-following paradigm.]

Therefore, we suggest the following approach: using the same training corpora, we assigned random sentences as priors to the model, and observed the change in surprisal only for the subsequent sentence.
We aligned the domain of the prior with the domain of the reader, such that a physics reader would receive a physics prior, regardless of the stimuli text.

We set the length of the prior to be a full sentence, capped at 128 tokens, approximately half of the length of the mean stimuli text. The total length of the prior + the stimuli did not surpass any model's context window.

To reduce variance, we repeated the process 20 times, and averaged the surprisal values for each word in the stimuli text.

The new surprisal values generated by the prompted model cannot be compared directly with the baseline model, since context lengths differs.
[!!!] Previous studies have found that longer context [affects] human reading-time fit negatively, with shorter context providing a better fit, we therefore expected the prompted models to provide a worse fit to the total fixation time.

To combat this confounding factor, we introduced a new pseudo-baseline, using our scraped "neutral" corpus, with no domain-specific data, we repeated the above process, only this time providing the same prior text to each reader, regardless of their domain. For this variant, the prior passages of the "neutral" corpus are unrelated to the stimuli text, essentially creating noise.
This allows us to compare the effect of a domain-aligned prompt to a neutral prompt, while keeping the context and prior length as constants.

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

[!!!] and the indicators were sum-coded

### methodology

We start by fitting a baseline model, with no surprisal term, and then fit a second model with the surprisal term included.
We then compare the two models using a likelihood ratio test, to determine whether the surprisal term improves the model fit.

We then introduce an aligned-surprisal term, where the surprisal is estimated by a model that has been fine-tuned on the same domain as the reader's domain of expertise. Such turn is calculated for each of the fine-tune checkpoints - steps of 4^n.
We then fit the linear mixed-effects model with all fine-tuned variants of the surprisal terms and the two prompt-based variants (aligned, and neutral),

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
