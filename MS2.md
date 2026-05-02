![][image1]

# Milestone 2 Report

**Team 9**

---

## Member 1
**Maria Ashraf Eskaros**  
ID: 55-7911  
[maria.eskaros@student.guc.edu.eg](mailto:maria.eskaros@student.guc.edu.eg)  
T-1

## Member 2
**Marina Samir Fahim**  
ID: 55-1622  
[marina.fahim@student.guc.edu.eg](mailto:marina.fahim@student.guc.edu.eg)  
T-1

## Member 3
**Malak Hesham Montasser**  
ID: 55-6656  
[malak.montasser@student.guc.edu.eg](mailto:malak.montasser@student.guc.edu.eg)  
T-2

---

**Supervised By:** Mayar Osama

---

## Note

The notebook **MS2_final_QA** represents the final version of the project and contains the complete implementation and results.

Other notebooks included in the submission were used for experimentation and intermediate trials during development.IV



## 1. Introduction

In this milestone, we developed a Question Answering (QA) system capable of processing structured triplets of the form:

(Question, Context, Answer)

Two architectures were implemented and compared:

- BiLSTM-based model
- Transformer-based model

Both models were trained from scratch without using pretrained embeddings. The goal was to understand model behavior, compare performance, and analyze the full data processing pipeline.

---

## 2. Dataset Analysis

The dataset consists of multiple JSON files containing QA pairs across different topics.

### Dataset Structure

The dataset is organized into two main types of files:

- **Training datasets (`*_qa_dataset.json`)**
- **Testing datasets (`*_test_set.json`)**

Each topic has:
- One dataset file used for training/validation
- One separate file reserved for testing

### Data Splitting

The provided QA dataset files were **not directly used entirely for training**. Instead, they were split programmatically into:

- Training set
- Validation set

This was done using a random split ensuring that:
- The model is trained on one portion of the data
- Hyperparameters are tuned using the validation set
- Final performance is evaluated on the separate test set files

### Observations

- Answers are not always exact substrings of the context
- Some answers require semantic understanding or paraphrasing
- There is variation in:
  - answer length
  - wording
  - structure


---

## 3. Design Choices

### 3.1 QA Formulation

The problem is treated as extractive QA:

Context → predict start and end positions

During preprocessing, it was observed that a large portion of the dataset could not be used because many answers were not exact substrings of the context. As a result, applying strict exact matching led to dropping more than half of the samples, which significantly reduced the effective training data.

To address this issue, a smart matching strategy was introduced instead of discarding these samples. The approach works as follows:

- First, exact matching is attempted to locate the answer span in the context.
- If exact matching fails, a TF-IDF–based similarity method is used to identify the most relevant span in the context that is semantically closest to the answer.

This approach allowed us to:
- Retain a much larger portion of the dataset
- Improve data efficiency
- Provide approximate supervision for samples where exact spans are unavailable

Despite the partial mismatch between answers and context, the task was kept as extractive QA rather than switching to generative QA. This decision was made for the following reasons:

- Extractive QA provides a more structured and stable learning objective compared to generative models, especially when training from scratch.
- The milestone requirements emphasize understanding model behavior and architecture rather than achieving perfect linguistic generation.
- Using extractive QA enables direct evaluation through start/end position prediction, which aligns well with the implemented Transformer encoder architecture.

As a result, the final formulation remains:

Question + Context → predict start and end positions

with a preprocessing step that improves label alignment using smart matching instead of discarding data.

---

### 3.2 Text Representation

Input format:

[Q] question [C] context

Target format:

<start> answer <end>

### Preprocessing:

- Arabic normalization
- punctuation removal
- whitespace normalization

---

### 3.3 Tokenization

Keras Tokenizer was used to convert the combined question-context input into numerical token IDs.

The input format is:

[Q] question [C] context

A vocabulary size of 10,000 words was used, with `[UNK]` representing unknown words.

All input sequences were padded or truncated to a fixed maximum length:

MAX_LEN = 384

The model uses:

- `y_start`: the start token position of the answer
- `y_end`: the end token position of the answer

Samples whose start or end positions exceeded `MAX_LEN` were removed to ensure valid training labels.

---

## 4. Model Architectures

### 4.1 BiLSTM Model

The BiLSTM model follows an encoder-only architecture designed for extractive question answering.

Encoder:
- Embedding layer to convert tokens into dense vectors
- Stacked Bidirectional LSTM layers to capture contextual information from both directions
- Dropout layers to reduce overfitting

Output:
- Two separate dense layers are used to predict:
  - Start position of the answer
  - End position of the answer
- Each output is a probability distribution over all positions in the input sequence using softmax

This model processes the entire input sequence (question + context) and learns to identify the most relevant span corresponding to the answer.

---

### 4.2 Transformer Model

The Transformer model also follows an encoder-only architecture.

Encoder:
- Token and positional embeddings to represent both content and order of tokens
- Multiple Transformer blocks consisting of:
  - Multi-head self-attention layers to capture relationships between all tokens in the sequence
  - Feed-forward neural networks for feature transformation
  - Layer normalization and residual connections for stable training

Output:
- Two output layers predict:
  - Start position of the answer
  - End position of the answer
- Each output is a probability distribution over sequence positions

Key advantage:
- Self-attention allows the model to capture long-range dependencies efficiently
- Better global understanding of the relationship between question and context compared to sequential models
---

## 6. Inference

During prediction, the model receives the normalized question and context in the same format used during training:

[Q] question [C] context

The input is tokenized and padded to the fixed maximum length. The model then outputs two probability distributions:

1. A probability distribution for the start position
2. A probability distribution for the end position

To avoid selecting tokens from the question part, positions before the context are masked out. Padding positions after the real input tokens are also ignored.

The final answer is extracted by:

1. Selecting the token with the highest start probability
2. Selecting the token with the highest end probability
3. Ensuring the end index is not before the start index
4. Returning the text span between the predicted start and end positions

Therefore, inference follows this pipeline:

Question + Context → Tokenization → Start/End Prediction → Span Extraction → Predicted Answer


---

## 7. Evaluation Metrics

Several QA-specific metrics were used to evaluate the predicted answers.

### 7.1 Exact Match (EM)

Exact Match measures whether the predicted answer is exactly identical to the ground truth answer after normalization.

- EM = 1 if the prediction exactly matches the true answer
- EM = 0 otherwise

This is a very strict metric, because even a small wording difference results in an incorrect prediction.

---

### 7.2 F1 Score

F1 Score measures token-level overlap between the predicted answer and the ground truth answer.

It is based on:

- Precision: how many predicted tokens are correct
- Recall: how many ground truth tokens were recovered

F1 is more flexible than EM because it gives partial credit when the predicted answer overlaps with the correct answer.

---

### 7.3 Semantic Accuracy

Since some answers may be semantically correct even if they do not exactly match the ground truth wording, an additional semantic evaluation metric was used.

A multilingual Sentence Transformer model was used:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

This model maps sentences and paragraphs into a dense vector space and can be used for semantic similarity tasks. Cosine similarity was then computed between the predicted answer embedding and the ground truth answer embedding. A prediction was considered semantically correct if the cosine similarity was greater than or equal to `0.70`. :contentReference[oaicite:0]{index=0}

The semantic evaluation reports:

- Semantic Accuracy: percentage of predictions with cosine similarity ≥ 0.70
- Average Cosine Similarity: average semantic similarity between predicted and true answers

---

### Why not rely only on normal accuracy?

The training accuracy reported by the model is token-position accuracy. It does not fully reflect answer correctness because a predicted answer may be semantically close to the ground truth even if it does not exactly match the same tokens.

Therefore, EM, F1, and semantic similarity were used together to provide a more complete evaluation.

## 8. Experimental Results

| Model        | EM (%) | F1 (%) | Semantic Accuracy (%) | Avg Cosine Similarity |
|--------------|--------|--------|------------------------|-----------------------|
| BiLSTM       | ~9.8   | ~49.2  | ~47.54          | ~0.614       |
| Transformer  | ~31.1  | ~50.5  | ~44.26           | ~0.631        |

---

## 9. Analysis

### 9.1 Model Comparison

BiLSTM:
- Captures sequential information
- Struggles with long dependencies
- Lower EM score

Transformer:
- Uses attention to model relationships
- Better context understanding
- Higher EM score

---


### 9.2 Effect of TF-IDF Matching

- Increased usable training samples
- Introduced some label noise
- Improved training stability

---

## 10. Limitations

1. Limited dataset size:
- The dataset is relatively small, which increases the risk of overfitting.
- The model may learn patterns specific to the training data and fail to generalize well to unseen examples, as reflected in the test set results.

2. Extractive QA limitation:
- The implemented models follow an extractive QA approach, where answers are predicted as start and end positions within the context.
- This approach requires the answer to exist explicitly in the context and be represented as an exact span.
- As a result, the model does not fully capture semantic meaning and cannot generate paraphrased or inferred answers.
- In cases where the correct answer is expressed differently from the context, the model may fail even if it understands the meaning.

3. Lack of semantic understanding:
- Although techniques such as TF-IDF matching were used to approximate answer spans, the model still relies on positional prediction rather than true language understanding.
- Compared to generative QA models, extractive models are less flexible in handling paraphrasing and implicit answers.

---

## 11. Conclusion

This project demonstrates the implementation of extractive question answering models using both BiLSTM and Transformer architectures.

An extractive QA approach was chosen because it is simpler to implement and more suitable for the available dataset and project scope. It allows the model to predict answer spans directly from the context using start and end positions, making the training process more stable compared to generative approaches.

The results show that Transformer models outperform BiLSTM models in QA tasks, due to their ability to capture global relationships within the input sequence through self-attention mechanisms.

However, the extractive formulation introduces limitations, as it requires answers to exist explicitly within the context and cannot handle paraphrasing or implicit reasoning effectively.

Overall, this project provides a solid baseline for question answering systems and highlights the trade-off between model simplicity and semantic understanding.


The system pipeline is:

Input → Preprocessing → Tokenization → Model → Answer → Evaluation

---

## 12. Future Work

Explore hybrid extractive + generative models:
  A hybrid approach could combine the strengths of extractive and generative QA. The extractive model can first identify the most relevant span in the context, while a generative model can refine or rephrase the extracted answer. This would allow the system to produce more natural and semantically accurate answers, especially when the exact answer is not explicitly present in the context.

[image1]: <data:image/png;base64,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>