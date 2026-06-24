= Project Progress Report: Correlation study between domain-experts and fine-tuned language models

Ely Raz - 209361153, Noam Peleg - 209515154
== Introduction
In this project, we propose to study the correlation between deep learning language models and human gaze tracking, specifically in the context of domain expertise. We aim to investigate whether fine-tuned language models exhibit patterns more closely aligned with domain experts' gaze patterns compared to novices.

== Methodology

We start by reproducing the results of previous studies, using the PoTeC dataset. This is a required baseline because of the change of environment: a deviation from the original comprehension reading task, and the change of languages from English to German.

Next, we will define the following three variations: (1) a baseline German model. (2) A baseline model with a preliminary prompt for the role of a domain-expert. (3) A fine-tuned language model on a domain-specific dataset.

For the fine-tuning process, we plan to use a source of German domain-specific texts (biology and physics) that are not present as stimuli in the PoTeC dataset. We are motivated by the idea of acquiring knowledge in a human-like manner, and so the new corpora used for fine-tuning would ideally be open-source undergrad and graduate level textbooks.

In order to evaluate similarity between the two groups of human readers and three models (base, prompted, fine-tuned) we will be examining the following:

- *Surprisal* Using both Pearson correlation and linear regression we will examine the relationship between each model's surprisal to reading times. We hypothesize that the surprisal of the fine-tuned model will align more closely to the domain experts, whilst the base model will be more closely aligned with novice readers.

- *Attention:* Using the three methods applied in (Mouratidi & Poesio, 2025) we will examine the correlation between attention activations and gaze patterns, comparing the three models to both groups of readers.

- *Full regression analysis:* We will attempt to explain a higher percentage of the reading-time variance using features like word-frequency, word-length, attention and surprisal, to model the reading-time of experts and novices alike, swapping the attention feature between the baseline and pre-trained models.

To conclude, we will employ statistical tests to compare the Pearson and regression coefficients, verifying the statistical significance of the results.

== Results
So far, we have examined the Pearson coefficient between base-GPT-2-surprisal and reading time, in the domain of physics.

The coefficient for expert-level readers is 0.657, while for novice-level readers it is 0.685 (_p_ < 0.001 for both). This is in line with our hypothesis, as the base model's surprisal is more closely aligned with novice readers than it is experts.
