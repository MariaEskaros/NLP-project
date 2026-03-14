# Milestone 1 – Arabic Dataset Understanding and Preparation

## 1. Overview

The goal of Milestone 1 is to analyze and prepare an Arabic dataset before applying neural models in later milestones. The dataset consists of raw Arabic transcripts from ElDa7ee7 Season 8 episodes together with extractive question–answer (QA) pairs derived from those transcripts.

The objective of this milestone is to understand the dataset structure, analyze linguistic patterns, detect noise and inconsistencies, and normalize the text so it can later be used for neural modeling.

---

# 2. Design Choices

## Dataset Exploration Strategy

Before applying any preprocessing, exploratory data analysis (EDA) was conducted to understand the dataset characteristics. The analysis focused on:

- distribution of question lengths
- distribution of answer lengths
- transcript length statistics
- word frequency patterns
- TF-IDF keyword analysis

These analyses were chosen because they reveal both **statistical properties of the dataset** and **linguistic characteristics of the text**.

## Word Frequency Analysis

Word frequency analysis was performed on the transcripts as well as the QA pairs. This helps identify commonly occurring words and conversational markers present in the spoken dialogue.

Since transcripts come from spoken content, frequent tokens are expected to include connectors, pronouns, and conversational expressions rather than purely informational vocabulary.

## TF-IDF Analysis

TF-IDF was applied to the transcripts to detect **topic-specific vocabulary**. While raw frequency highlights commonly used words, TF-IDF emphasizes words that are particularly important within a specific transcript compared to the rest of the corpus.

This method helps reveal the main topics discussed in each episode.

## Arabic Text Normalization

A normalization pipeline was implemented to clean the raw transcripts before modeling. The preprocessing pipeline performs the following steps:

- removal of timestamps
- removal of Arabic diacritics
- normalization of letter variants (e.g., different forms of Alef)
- removal of special characters and formatting artifacts
- collapsing repeated characters used for elongation

These steps were chosen because subtitle-based transcripts often contain formatting artifacts and inconsistent spelling that increase vocabulary size and reduce model efficiency.

---

# 3. Output Analysis

## Question Length Distribution

The average question length is approximately **6 words**. The distribution shows that most questions are relatively short. This indicates that the dataset focuses on retrieving specific pieces of information from the transcripts rather than requiring long explanatory answers.

## Answer Length Distribution

The average answer length is approximately **5 words**, which confirms that the dataset follows an **extractive question answering format**. Answers correspond to short text spans taken directly from the transcripts.

## Transcript Length Distribution

Each transcript is relatively long, with an average length of approximately **6,875 words**. This suggests that the QA task requires locating small answer spans within long documents, which can introduce retrieval challenges for machine learning models.

## Word Frequency Patterns

Word frequency analysis shows that the most common tokens include conversational Arabic words and connectors. These words appear frequently because the transcripts represent spoken dialogue rather than formal written text.

The results also reveal the presence of **Egyptian dialect expressions**, which reflect the conversational style of ElDa7ee7 episodes.

## TF-IDF Keywords

TF-IDF analysis highlights words that are particularly important within each transcript. These keywords reflect the main themes discussed in each episode and help identify topic-specific vocabulary.

Unlike simple frequency analysis, TF-IDF suppresses common stopwords and emphasizes informative terms that characterize each document.

---

# 4. Insights

The exploratory analysis reveals several important linguistic characteristics of the dataset.

First, the transcripts contain a mixture of **Modern Standard Arabic and Egyptian dialect**, reflecting the conversational style of the show.

Second, the dataset includes instances of **Arabic–English code-switching**, particularly when discussing scientific or technical concepts.

Third, the transcripts follow a **dialogue-driven narrative structure**, where sentences are relatively short and appear as conversational segments rather than structured paragraphs.

Finally, the presence of timestamps and formatting artifacts confirms that the transcripts originate from **subtitle-based web-scraped data**, which requires preprocessing before being used in NLP systems.

These observations highlight the challenges of working with real-world Arabic data, particularly when it contains dialectal variation and informal speech patterns.

---

# 5. Limitations

Despite the preprocessing and analysis performed in this milestone, several limitations remain.

First, the normalization process simplifies orthographic variations in Arabic, which may remove certain linguistic distinctions.

Second, the dataset size is relatively small, consisting of only 13 transcripts. This may limit the generalization ability of models trained from scratch in later milestones.

Third, the conversational nature of the transcripts introduces irregular sentence boundaries, which may affect sequence-based models.

Finally, dialectal expressions present in the transcripts may introduce additional complexity when training models that assume standard Arabic structure.

These limitations highlight the challenges associated with modeling conversational Arabic text and will need to be considered when designing neural architectures in later milestones.