![][image1]

# Milestone 1 Report

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
**Malak Hesham Montasse**  
ID: 55-6656  
[malak.montasser@student.guc.edu.eg](mailto:malak.montasser@student.guc.edu.eg)  
T-2

---

**Supervised By:** Mayar Osama

---
# Milestone 3 — Retrieval Augmented Generation (RAG) System Report

## Project Overview

This project implements an Arabic Retrieval Augmented Generation (RAG) system capable of:

* Processing Arabic transcript datasets
* Normalizing and chunking text data
* Generating semantic embeddings
* Storing embeddings in a vector database
* Retrieving relevant context for user queries
* Generating grounded answers using Large Language Models (LLMs)
* Supporting memory-aware conversations
* Supporting semantic caching
* Detecting out-of-domain questions
* Evaluating model performance using multiple metrics

The notebook demonstrates the complete end-to-end RAG pipeline implementation and evaluation.

---

# System Pipeline

## 1. Transcript Normalization

Arabic transcripts are first normalized before retrieval and embedding generation.

Normalization helps improve:

* Token consistency
* Semantic retrieval quality
* Embedding robustness
* Search accuracy

The notebook imports:

```python
from nomralization import normalize_arabic_for_rag
```

Normalization is applied to transcript files before chunking.

---

## 2. Text Chunking

The normalized transcripts are divided into smaller semantic chunks.

Chunking is important because:

* LLMs have context length limitations
* Smaller chunks improve retrieval precision
* Embeddings become more semantically focused

The notebook uses:

```python
from chunking import build_chunks_from_folder
```

Each chunk stores:

* chunk text
* metadata
* source information
* transcript origin

---

## 3. Embedding Generation

Embeddings transform text into dense semantic vectors.

The notebook loads an embedding model:

```python
from embeddings import EmbeddingModel
```

Embeddings are generated for:

* transcript chunks
* user queries

This enables semantic similarity search instead of keyword matching.

---

## 4. Vector Database (ChromaDB)

Embeddings are stored inside a Chroma vector database.

The notebook initializes:

```python
from vector_store import ChromaVectorStore
```

The vector database is responsible for:

* persistent embedding storage
* semantic nearest-neighbor retrieval
* efficient similarity search

The notebook checks whether embeddings already exist before re-adding them.

---

## 5. Semantic Retrieval

The retriever searches the vector database for the most relevant chunks.

The notebook imports:

```python
from retriever import Retriever
```

The retriever:

* embeds the query
* compares embeddings semantically
* returns top-k relevant chunks

The notebook tests:

* Arabic queries
* English queries
* mixed-language queries

Example tested query:

```python
"الأخطبوط ذكي ازاي؟"
```

---

# Prompt Engineering

The project implements multiple prompt construction strategies.

The notebook imports:

```python
from prompts import (
    build_minimal_prompt,
    build_strict_grounded_prompt,
    build_arabic_strict_prompt
)
```

## Prompt Types

### Minimal Prompt

A lightweight prompt containing:

* retrieved chunks
* user question

### Strict Grounded Prompt

A stricter prompt designed to:

* reduce hallucinations
* enforce grounding in retrieved context
* improve answer reliability

### Arabic Strict Prompt

An Arabic-specialized grounded prompt for Arabic answer generation.

---

# LLM Integration

The notebook integrates multiple LLM providers.

## Groq

```python
from llm_client_groq import GroqLLM
```

## HuggingFace

```python
from llm_client_hf import HuggingFaceLLM
```


# RAG Pipeline Architecture

The notebook connects:

* Retriever
* Prompt Builder
* LLM
* Memory
* Cache

through the custom RAG pipeline.

The pipeline is imported using:

```python
from rag_pipeline_ import RAGPipeline
```

The pipeline performs:

1. Cache lookup
2. Retrieval
3. Out-of-domain detection
4. Prompt construction
5. LLM generation
6. Memory updates
7. Cache storage

---

# Conversation Memory

The project implements memory-aware conversations using:

```python
from conversation_memory import ConversationMemory
```

## Implemented Memory Strategies

### A) Full History

Stores the entire conversation history.

Advantages:

* maximum context retention
* better long-term coherence

Disadvantages:

* larger prompts
* higher token usage

---

### B) Sliding Window

Keeps only the most recent conversation turns.

Advantages:

* reduced prompt size
* lower token consumption
* faster inference

Disadvantages:

* older context may be forgotten

---

### C) Strict Truncation

Restricts memory using character limits.

Advantages:

* strict control over prompt length
* prevents context overflow

Disadvantages:

* possible information loss

---

### D) Summarized Memory

Conversation history is summarized before reuse.

Advantages:

* smaller prompts
* preserves important context
* efficient memory compression

Disadvantages:

* summarization may omit details

---

# Semantic Cache

The project includes semantic caching to avoid repeated LLM generation.

Cache workflow:

1. Incoming question is embedded
2. Similar previous questions are searched
3. If similarity exceeds threshold:

   * cached answer is returned
4. Otherwise:

   * normal RAG pipeline executes

Benefits:

* lower latency
* reduced API cost
* faster responses
* reduced repeated computation

The notebook demonstrates cache hits and cache misses.
The following figure demonstrates a successful semantic cache hit where a previously answered question was retrieved directly from the cache instead of triggering a new LLM generation request.

![Semantic Cache Hit](images/cache%20hit.jpeg)

---

# Out-of-Domain Detection

The system includes retrieval-score-based out-of-domain detection.

If the best retrieval score is below a threshold:

```python
out_of_domain_threshold = 0.45
```

then the system returns:

```text
السؤال خارج نطاق المحتوى المتاح.
```

Benefits:

* reduces hallucinations
* prevents unsupported answers
* improves system reliability

The following example demonstrates the system detecting a question outside the supported knowledge domain and safely returning an out-of-domain response instead of generating unsupported information.

![Out-of-Domain Detection](images/English%20%2B%20out%20of%20domain.jpeg)

---

# Evaluation Framework

The notebook evaluates multiple models using several metrics.

## Evaluation Metrics

### 1. BERTScore

Measures semantic similarity using contextual embeddings.

Higher values indicate:

* semantically similar answers
* better generated quality

---

### 2. Semantic Similarity

Computed using cosine similarity between embeddings.

Range:

* +1 → highly similar
* 0 → unrelated
* negative → semantically dissimilar

---

### 3. Grounding Score

Measures how much of the generated answer appears in retrieved context.

Higher grounding indicates:

* lower hallucination
* stronger retrieval grounding

---

### 4. Latency

Measures response generation speed.

Lower latency is preferred.

---

# Evaluation Results

## Model Comparison

| Model            | BERTScore | Semantic Similarity | Grounding Score | Latency (s) |
| ---------------- | --------- | ------------------- | --------------- | ----------- |
| Groq             | 0.6652    | 0.1756              | 0.1632          | 0.4492      |
| HuggingFace_Qwen | 0.5382    | -0.0144             | 0.0077          | 14.9627     |

---

# Observations

## Groq Performance

Groq achieved:

* higher semantic similarity
* higher grounding score
* lower latency
* stronger overall generation quality

The Groq model produced:

* more relevant answers
* better contextual grounding
* faster responses

---

## HuggingFace Performance

The local HuggingFace model showed:

* lower semantic similarity
* lower grounding quality
* significantly higher latency

The negative semantic similarity value suggests:

* weak semantic alignment with reference answers
* possible hallucination
* low retrieval grounding
* unstable local generation

The slower inference is expected because the model was executed locally on CPU.

---

# Challenges Encountered

During implementation, several technical issues were encountered:

## API Rate Limits

Groq API limits caused temporary generation failures due to token quotas.

---


## Local CPU Inference Limitations

Local HuggingFace inference experienced:

* high latency
* weaker generation quality
* slower evaluation

---

## Evaluation Stability

Evaluation metrics changed across reruns because:

* generation is non-deterministic
* cache behavior changed
* retrieval order changed
* different answers were generated

---

# Conclusion

This project successfully implemented a complete Arabic RAG system with:

* transcript preprocessing
* semantic chunking
* embedding generation
* vector retrieval
* grounded prompting
* memory-aware conversations
* semantic caching
* out-of-domain detection
* multi-model evaluation

The evaluation demonstrated that:

* Groq significantly outperformed the local HuggingFace setup
* grounding-aware prompting improved answer quality
* semantic caching reduced repeated computation
* retrieval quality strongly impacted final generation performance

The project provides a strong foundation for scalable Arabic conversational AI systems using Retrieval Augmented Generation.

---

# Future Improvements

Potential future improvements include:

* GPU-based local inference
* hybrid retrieval methods
* reranking models
* improved Arabic embeddings
* multilingual support
* adaptive chunking
* better hallucination detection
* larger evaluation datasets
* retrieval reranking pipelines
* advanced conversational memory compression


---

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUIAAACdCAIAAABO0OaFAAAtIElEQVR4Xu1dTYgdx7ltHo5MdrOLxRBkmJfYGCcrC0SEwywykJWslccgDNYmk8UQwgTmLWyexwJvhKVFtLOSxTiGDCEQFBDhbbISAZHNeCECxnlj2UFKJFkOepEzkq/V73Sd29/U1F9Xd1f33HvVh2K4c291dXXVd76fquqqLB8wYMCUIzO/GDBgwLRhoPGAAVOPgcYDBkw9BhoPGDD1GGg8YMDUY6DxgAFTj4HGAwZMPQYaDxgw9RhoPGDA1OOxpPGjfKR9Nj8MGDBtmGka72cmqbtH4BLyjf3TgAFTgVmm8ahM8u+X9z5D+uf2n+/96Yok/Isvv8q/2Ms//ZZ5pMH8bUC/QBfcv3//2rVrV65cuXz58tbW1ubm5sVf/BIJHy5duoQvr169igzI1qy/ZorGYm/HH0ajf3/8EYh669w7f//xj26ffOlvT33jRpY50+2nv13kef9dXDItZMYDXr9+HcIBaVhfXz99+vTS0tLzz3/n0JNfzzQ89dRTR48ePXHixMrKypkzZyBGuOTOnTvNJMYGitr2ANUzc3cG0MC8vQIYsrv7wMzdMfDg4Cc6Bc2u90UM0Ilra2voJjxRZB/NCI3FkMKugodg4821n9w49iIpejObu334SJGe/nYoHT6CnMUlr7wMEz2ZTEa/onehyJeXl83+1/DE155kMn/QMDc3h0IuXLgAWY8UFycgr2bRJVC4mbszLC4umrcvAUVj5u4AaEOojI2NjWeefc6oADoCujWQJINxIQDtDHMNQ23eT8NU0liMrYge2Pt///OHm2+8/sm3vrvHW5uokenwkcI+v/2WGPZJAMgG7Q7TqvdxpXwEklHO6uoqrLR51whAyFCCs3yoGzN3Z6Ddc1ajaxrD9p49e1bvmjb9oicpEICJ3tnZMe+tMA00Nkyi9i/Ye+d3v7752qskXsFem5PNkioQbnZ+0Ey+e/cuyLCwsCDdmUpE9CSFw0TXJfPjTGN4RghVpPUOaXY1baJjdf78ebMGCtNAYwXaXjIKoQ4iXnAMsW5bwxtMKBzO+UHZZBAYOl4XkR4SbgSzb1YliMeTxihTCNwRdfVEGk+tNVZzvEKkIu59+y16zt2xV0+4EQy+UakeAAKU/HWIZkeJtzOrUoXHkMbSOxyAsG8aTjJyEX8t7jU/P2/Wo0TtPusN+ggTPiP0vX3ypcSec0w6fARag9NR/QB+moyR2N3pS9TWlQjLDTIg/DYrVIXHisYIg48ePVrZkkyRnWLDLgpfnjlzxqxNiQmjscZb4v4/Pr517p0+za+dcGsokX5ojOCHHRmWEiryfT2fZYuLi6urq+hsFHLhwgVQCH/xGZ75xsbG6dOnn3/+O8YlhkHAN3UD4/xxojGf1HkXSXbXwPdGR6BhoaBRE2P2C8EiZ+yQAdnQg/q10jv4fPXqVf1CHZNEY22N5Ej5z4hLi+i0Z/Nrp8NHbr7xul7TLnD//n0KYiWBpY+XlpbQ8ehdRNFmcX4gMy7BhfqUpgifmTsCjwmNqWHDvSPtia7Z2tpqPG2OCqNV9cGzLNg1od96xaNCLZHJ/9z+899//CPTfz4gU1ykw0duvPKyWeGkQLch8slcUihJCAx5rZxIjAQsA4yAiAsstpkjAo8DjdfW1pyF63chwPb2txOAFGje48ePo4/M3zRMBo3LcSwQGIQxCXzgqWMa7+zsUAJs4WASAkOYGiv4MKAUYJ8hMeYPEZh5GkdyGA/bZhVNGGGH6yBprA9i3fvTlZuvvdrp7FHz1CWNQUsKgS0cuohAGXdE4PaYbRqfOXPGWawUzt5J4hw1xkHSWFxokKQJgbm+Uq2gZPrbU98Yf2lnbpNQ4NtvmZVPAahYLv2x5UOkpMF6jJ4xwzTe3Nx0liklA81cmLTol8bK9so88L8//ggxcD0Cl8uei78vHLtx7MXChr/x+q1z7yCBbPgXX95Ww8upPPOinD/81nyW1oADhpjHJyV0pJeXlw9WzcdgVmm8vb2d+ce08NPCwoJvPUbP6JXG8jLgl/c+A/eiCFza22LCCYxa+8ntzV/BA5dXCw1I+f+69hfk/ORb321J5o+++TQqgALNO7XG+vq6U+xESjY2NsxrJhIzSWNoT5+jRA179OjRcLzaJ3qisYTB4N7t998FPQoH2OKMkcheuNzj9wfV+IGTuua6a4WC0qMR9AVVQLNUuOtqPWYB112awSf6Inlnz541r5lU+J5lqmkcGNaiHZ4oL6kPGpNmDIM/+eEPqhdyqNcSkPPub96//4+PxQm3CRwG8+NvoQjCd/Qn1AQaxCi5JaBcMgVbRCgl02KHidmj8dWrVzOPO82Om7Thxs5pTPqBjZwKDtFJsbew0m+/BZe4GXWdgBPezCAXpriDhR8BdzpT08LmBZON2aMxl1vaRbG0CRxx7JbGJCFiVJAz5EWX5vfO7369N/OWzoMtBsMb0RiBMaLiJKpEwBmmgKafKG8tBjNGY+6CYJfDkLjBmvMe0AmNC7lXXjTc0bEXbTFknBSB4fTCYPLCLibQb517p8FA1w1tKXXdd/cCOHXqVEBKJlDTV2LGaMxXu53lAP3vBxSDDmisCPxV/sVnF38e8qLV+DM8bW6Xs+dCpzPCeVkmVIl596pU0F7t/pGrV47Qf0n0Cxds+Uxxs7WQB45ZojHUqLMQKtmtrS3zgslAYhrz5f4KI8ydOl57dW/zui6gFmmP1BuO3pr4klq2xYpB+1I9x4tCAL6omFIyaQMnkZglGgcKmZubM3NPDJLRWMxpsZ2db0K4dKFhgceXJbW9Jh4VTkGxGsRZGV86fAQhsQSo8tpAe78aZbIop5Ssrq6aF0wJZobGyJb5faXNzU3zgolBChqXa7NgXflig5M2BYGPvQjbyIu6MsIlqFPGi0ysyoQT3YS8XItHU9l+zZ1P3Fk+XHfzgimB77mmjsYX1Z4edgnsoMmMionWNC5fTgI/xztjWay4ydXO779LL7egPVPHqD3PpLSPcJgh8RPli7jtJRKhr1NK8OXx48fN3NODmaGxb21sNvHDFi1oXPKwWJj19ltuwnAq+Kc/k1UcPYA3KnYd8Pn2zlRyeFzIaET2Co1bzjRAl2cKTinpU9yTYzZozJDH51G398U6RVMalzt1gJ+3T77kM8KffOu7MIljAsvmHh3b4ZGqFW5di8M3XjiGq6QQmd0VGrdclcF19raIsPAJWWHfDLNBY98YNUuYnOXTTjSkMZlZvCTssXgwwrfOvTN+e6EXF7qAusuX9z4DJ521MhJfeyhWbh97kS8/iL9A46nTeGlpSe7TAL64K1M7HiaZzToozAaNuZew8/KFhQUz94ShJo217WbH08IGNxQrYAllNrhPjMjhYy86vQNnKjj8ysvOpWMGjbN2U8eBwPjUqVNm7qnCbNA4cPna2pqZe8JQh8b0itUsTnFCks1hZYQRCY+p3kLoG0A4HGOHpbY333jdp25I3VQ09i3Tzfo96KgLzAaN2cXOy/t8imaIprFyjMdU4WYd+ykx3nnjD78l1XsGK+aL0h1Jjb2F95Fnv+oGuXGABBedAuGUkkuXLpkXTBVmgMYSQzkvn/wVsrE0pskqZoaVy1oMIGmsKL754Q/Gq7JGxQaX/UEpF45pRXKYL2lwCUrAvNo0bjy1G15X0H5hycFiBmgcXiTbuN97Qw0aQ+4/+ubT4zEtOq7q7w11ZNl4TrhPlMtO/nXtL/Hj0kUscPKlmAkwdmESGnPc2xYR3iL5MPXdRDDL9WAGaMw1Ava1vHzyF8nWoPHd37x/87VXwVikYsn0sReL9MKxW+fe2eNDP3aYd1F/7/3pCsfVbMaaSTnSrG2YwARPYNFpHNi2PwyflLDYSiGrCwpfS8QvIZ4BGnOfAPtadtDkvzoaS2N9jFqH88tuUdYkp2ZRBC6mjmzeakmfxB6P1VWBR6XoNG68BiA8aVwpZHVhKKAGCZfj8c1yPZgBGvsmjdkU/TuadVFFY0757uew/pf4Kv9C+69LlJUpRsu5yVaEL023f7wtnmbJw+DSPF2yG49FHQiN7dvFp4HGTFNOY+HtaPRodGu0+8GDe7/X08P7f8T3SHsLPLRru0JZK99ouSMdPlIMaKnxc1FDkVhaWkpFY59TTSFLHnoNNDaqUUnjsFM9yS9FEC4aq5lhcHX3xqu7f/3Pr/43y//6H/jLhM9M/BcZHu58Hzkf3LmIS0a7H+b7CTP+kILbHEXbG2mzSbs/Fe9UvfKyvOpQFzaNG781Hh7iajxy5sNAY6MalTT2uUuRlx84TBoXDNz94Mud7wlRhbd6Em7bX4LYYPXD2+dgsfe2klaqgR8aQGypBMMVSY1m3d78VV0LrEOOsRUaNxbK8IRT45Dbh1qxsV0l1uqxorG8zea8PLmeTY59NIbEP/z8Pd3Y2gSOSXJhYa6VoYZq2LtNMyaPRn/76c+cS8eMRCMsa6Qb09geqW683Cq8YUDy99Hn5uZ4u0g4azUbNGbHVU6ehd2laVr+QTvsNLO1ks5h/XtY6S/+/l800XvGOQIjLjt54VilHeZwtL4zQcHhRloD4KEBOo0Dx71XguPeTilJvmT3ejR2dnacO8hNI40XFxd91aik8UjtHO6zxn0+RTPso/GDT0/atEybaOTBZwTSxeBZgGzlgBZ84/GaE4u3ksY7Frz9FuPnsQW2y6yDTEGncRu+LS8v+4TsYPcMaE9j3zBvpg77NXN3hsCq9ZgxKna38/I2/d4PNBrvftjGCMck3VAzFS634vO4Eop4NNQjtdJ1vCuQxdu9VO6wOR7Kip5PqgT7NRWNeVa9LSWRXl9HgNJrT2PfMG/Weq+FWuAdjTqwec2sLgT07Pz8vJl7wrD3hKCTTbzukkHpPX9bi2ZH3NfSd3A5XxV+4RiPO+RVcm17pKWxz2RRUJKPckUiCY25INn5XH3ufcPOsuuAv2ZWF3zvG7OQykGyg8UejUEknV0Hkgr7fOtNGQ8T9/if23+Gw7xvry81IfzZxZ/rC+VifKd4pKVxeLD6oLbFTELj655zMLLWW6bEw7cFTxYds4T1bOO5xn6wR2P4tySSwSubbMkT70KzzG8e7nz/4efvMXjOSzIXryKqs075VrO+505entmRcPmrzuH2NM6DwVvamscjCY0REfgohML7WQLV3iPwPQULabn3S9cwaTwJSdcdqBWNMy1zkUaj8blKZQAMPcpx4CxdkClDlwlp7AuPKSgHou+T0BhA9GgXwkZL1SNh+GxpVud4Sp+eZTmT7FdrNL71ps2o/pNumYXPxVqx0jjruv3atWucLSTlEgqN+MAJaRxeY3AgGz6lonFgsqeft6kvXLjgq0D8ElpfISynzXRj19ijMRd+9ONFB5JeAYPM48hZOWmgGc8008mWpXt310nj9uOugWWStQQuFVLRGCJuF8Jyki9ucYIrZ427s9fi9QiDfPsppKgDCXxioE84fWCwqGUSuyr/2sm+yk56Nphl+tJUnDrN2NCNXwk2INtBpKUxj6GwpeSQkvis95dpUtEYEYFdCMtZXl42c6eGbwse9lqtJvW5FSwq3j/vGXs0/ir/Ip5avmRfzhcn5C0Lg9V2CXbSs6E0Tkpx5ZBN41QGjbGWUXj71Ty+AVWmLIWmqIVUNPZtgsN269qIBQLjukPlvhVp8iyp3L200GbGH+UgiUGbWklsLCeB5TVGGY7Cv+q1xw/x68Pb58BtuSTAav374p0NBZtmbOXGy54N2GoiS+Qf+vxPkbw+V/CmonEeXAXV9aw4j8tz3rqBPHBFul0aCzx69Kh5wQRg32JMeS+ibhISPvj0JFjqdWLKsWUZdi54vvsB7osLdbr6WF3QWBXClUM2jSNnFyohh7DphScZTK40yFmPg6IJaby6umqXw6I6na3xTcizyxoYTwYIdoHyOBPoWu9bp1bYTxd5IlNhXeXNxBhoSyYLVu9+yJecWQfbUI8/K6faR+PIuf5KcE1PFzTO/SdIiPDNz8937YgSCWks/ovzibp718+3+iprYTmdbaKXnEoSUsFcbsq1XA0SQt+9pdHNoG1nD6/b4LN8II1l5ZDBtCxuHXwlaFuMwlMF3rlnolUXFLAo1eRZAAlpLDPtzsepG6NGgqbYrj9v2jgI8gXbTBSGriOFWjBpzPHqBgmxbg07XAUWBb3w4M7FYljLorG4pjaTr6fYE8fY+iN5zwXmkEUKQbAkzxJAQhoD6+vrdlFSYMLWEwSO1MnaDa35YgQmysPk2GSTxnnT1xW5fU9KlC63ip8/3L31Jse6afPtVVZCtiQ2k3JglJxqNosILDZgYh26kH5BWhr7VkTKs6T1LwJBbNZ6tQZUAF84twtnoki0vEsqOGjcwCDTSNaaoIuHPtBdbGxQ7jdAybBp3H4EwqkjskbjJWFwBYstIpL4jCsrK2kJIEhL49z/uh/LbBys2pDBEeeNshQqo9JjolQsLi527TRVwkVjZZDpx8an7mgsMEpneGmTDc26P2Nt2Gs/WHLycRq0mG+9gSErmZq1Tt7CyWkcMMgsFs/b/ikYu/pulLWIig3wfZsAk3k74OzZs0kGZZrBojH33FBbCNRicmieKSG0wW2aMoNsFPqWDWpPGrPY9greBpy3wIp8Q1YyNRGasBpglNN1zJrSGIA3ZBeol9wy5mcw4rtFlkKP6+C0QpjJompRtzYBuRPoI4QP4dNzLRqXTmzd148ffv5eHzTWYA8mC99aBrH2Sk92Ukvt4AOKlRE1W0T0xGpkauAXXduGDPAsOFXjvG/WgsYQu8A4vDTm+fPn65rl7e1t2a6URdnVzjqYdY9hstw9U0FQsYan3RY0UNYwJ1zZQgSay0Fjou4ccvGeY7/ge39OGrfcAorjn0aZc3NzgXZsD9FKtnwYSZhAoKehd9DlkHIQ27AGkAaINUgLwYKLeObMGYavArv8Q+1onJeuta9w/RGgSipDFTwCas5DPAJl8teO1sDRuw7cnUmkhYC2RdegSjEKF92EHoR2Xl9f5ys0hJQZeMfDS+MiQt6/rU+Y1TKG3BtkoNJIWetFIFyOZ5TZ6VIkgk+UVcmKiIsuMbXwhMeaScra0Tj3rwYx7iKA6oSKQUx7WQFNAV0M1fZ8+SZ5FmwWZkgySeEDt6QPV0NPdu/ASYHDDzW6UgKf8Q3jGgNSCD9kwYWlXhrnislwrcPsFQ7j7+7t/27pSNQCm1Unmy7cjUMU+xVFFtjRAgYDUNsc9GIdbOGISVJt+6fIlLWmcR5nwaR5A6h8FmbrlMMEhEpej7WrEU7hxww/4KGqmL+Cxl/lXxTvBkaYYmbgQFenzqfAuZBLxKJxp/peumg/jxUPMcsxHZw2yfPCRJjVqg8ZUu7oKVjVLOhwJgecBdnQ365SR4lP6jNOIRoTsgd9OI0Ncr8RMuXDTvg+PLIXAN9AsgsMuDRdYHf3Ae54qDQ1HdGAiYXzRgScvVTLThAnc1wqSy30LBMdnXDoPhLsHWkuu27JE4XQF/lH0BhB8p2LOledSX4qDmTjld072M6pYyFes4FlBmN2aanEuhbwCHArOI5NpOWzFEsgQMXtumAFB3sJuxp1E8tBT/nEuh+goUBmscxpu4aJskegd3wLkKpoXJ7SEhkkk8+9jXVx0NXmcGPi+Rz1rPUkVkugYmCCbDwmqCU6ukwIOJoKp7TrUAhCL1NcWc2aG5VHlNigczsC2g3ahLMbgrpPpyejm9BBm5ubYd1aReMSIHPMWmtSvdh+IO/DGjsdYCFeg+1j7NeMpbSYOYMeAPsMhQJKr6ysiLMaD1gwCBzoBBpAtXdNXRu4I25tCL1Ab3Abx48fP3/+vM8iHTgQuILP6+vr+ui6AUOofAB10UcozRcMG4igcclG2FgOd4UTmQzrzWUknULWyjlTVn8lgP1iE8uB49TMRe8B0NOcFgY4VUOgcfAvvgTtIfqRAtEnUG3Uc2NjA6y2RX9hYQGKGKyAzooX6AkBaotmx9PB0qyurkKunLNKeGowdm1tDdmQOXKG2UYEjTUUbxqVB5cHEjPsBck1EX+Vc/MAnX611oHIUni7nJYT0QMGdIoaNCa7yGThqjPxJ/xtwmQVjTMgr7zW99ax0C8LLmEzwNdlneWk2htowIAuUIPGRDEtHPHihPz68P4fY2lUApfkau7KucW8AfoqNveEgVtx73bLPhLOQnqebRowoBZq0zgfM7l6Mllscl0mczm3lP/g05MP7lzkm8aMt/XS7PXPBgMj10IjPvGVk/nn6wYMmAQ0oTERcx6y6V1Hj3hxq10jwZmX81OFl/L2iS/F2NLAuk6W0GzgYcCAftCCxsomG5vI22mPybCoZhluINvD2+ecRel6gZnDo1wkYZiHd+/eDXjmmVquMLHD1AMG5G1onCu+Vc5C6dwrTmDilUGzXNBYhceBVLwOrQq5W55nGUiZ2j7G6VqDn3wDzr5Kru3h3aYBA9qgKY01HoLJTh84t0azCyv66cmYNV6Vo2hf7nxPSOlcPmmzcXl52WDytWvXuIWNnV+/sNas1YAB/aMpjXWoKaLKDa6FlrDe8MbHobLPLKNMP435k+yt55sosgkJwnP10pUrV2RfBTuncdUwvjVgwtGaxhoVEf2K/+wknvxUmGUVKtM4CiEFI3Vsul2OngrHW0HeUK9M5K3AzmCnrP5SsAEDekZrGpcYKYdVn4gKkzAvD3xy2+RHewdK+cqRSNv5on+SlCXdk3XAgI6QjMYEB730UNlHQvn14e1znEDaC1vL3Tnt/HoqXtUo+R8epmqcsl4C493dB9sKU2H2oa6vX79uvHBzR0H/ZkCuBl+uXr3aQ8skpjGBnoapFKLaDNQTMiBa5gRSwV4FFiKuuDN9xTciFZN92+u1TFnH20qAwAzsBQf7OmQMOMe+srKif8nK69/4sLW1tby83PV7Dpubm6dOnWp5l42NjcbHTev7t2RVcRmiwqWlpXCeMKKavgHoYMvq6wAbJQPJLEs7GB7zV/sSJpk9Dq/fMBJf6axMmZoxdk5TJQGEjBsgwm+/cOECJW/yJ6j5Asna2pr+Jfdv0L/xgWLdtbbiQgB9bDL8vq4NTmRmjTYnl41BYF3QrUZb2eDua5Grhp2IavrG0EewA2SWn0jmIuhVe3px1xHfVbm8DqlMd2AnkGYps4Q1LbijbWN9f1BoSWMI6+nTp1vayUrgLvAX5C58L31/lmqcPXu2wRFNbJ+5ubl46wp1Aw8lPr+N2s9WD8rjLU45jTDLxk8IsMMbFZDzMsptn0jcMmVdGg1uM9J+98n+0YbG3bk2YXCfHfPbbsBF/j1vTtLHs424w+atN4WoYT4bXA3nLA6OU/C9LdwsUaHuf46UoN8V9qNgTC7+4pcw17pM4DHhp+GnnZ0d/MQ9A/ANx8lgPeDIUa8jA+2Jvp87iATdhDy4tri7NkfA19b5YfyrC5U05mYAUh/UQV8Ji1vgV26CL3ckUGE8iNQWV6GeCFD14Qlezg/4iXoBGfikUho+sJVwl0uXLpHG+MBb84PEL/iAf23icQ8GuQXvi9bTG9kAisrUpqJOhYUHZIfir+6ro3DUlq2EYvEZObnnkTw7nxf3dS4r7oPGgiJaVuFuIIl3LR/sPHrOwvEuISeDtE9ZxAsVbcBdtZxdQqA7D2lb3qFH+T23GZItoFHClhpN0be5woUcLBAINzjHLtA3387UslN9uy+nw19JY9aQfqxApJbGiruRZGqLDymEl1BwuVReIPxkJMJHztQdpSnkm7w83+uOOodB/zVTt2YdhCF8IvthucKPhGTz6g8F1WCHBqy2Mf5HSLAtkDZhsXxGVhg859pEKhe9czPXkYC90pgAmWVGSmdpgLHOxPzFIealX83etTnZIOmt3AXYH4EBLQoit8vi1kJU/5RaEPWZZ5+jSRE5hgwhs2xzBTrhX5p9SD+LxUMx2255cJQEDpSb48eP08KzELsRKmks9YFc4i5cLSc6kf9yPy1WQG7BWUN+BknwgDSnLI3f85y3rNyCl7oANccH2n9mYyPgWpJQqkfiUU7k1EX+a6/Vo1VgCbKzJ5oLhXAvR9th4bP7DAAqzB4nLWUuk/+yI3bUuTl0H1DDO+rgnqzcuIKf7bUMvdK4aA/lxRUb9N37ffidivg0XtpZbgaStWZyZolpcrCr+Bldi867rkCuUnOLfbiiNm2n0Ig5Femn6EiFqc7BAf7LNrE7Pi+LgoDyX0qtMIEvYOsOLVFJYxYr+6UwvxgoncacJiR/RupMae6CyOeVAwOYjSaINJY9lTjEYJ8EIDTmv0bozqukhizTVlg6jeliSI/Q6toGnGwXBeEDdZNs+ajTmHXLrI5gi+WlP2IY5F5pvG+1llqJrZO5rjWWS8avQKabQM6C7m4SZNqxDLrjR8kgE8Q+6KzmT7oAkcYiOpQDoQ2kEPpC910FvK8MxlJqxVGkGbcjxkgai3YwRFanMclA28VieRWdTCEhCySrSTkxgyzcHikM0zhXDrO0CS53to9NY7mv0ciCmCGPvAyhRRnZNNb3fjPG59jdxi36pbGFkUogs+8dqZikj1e3N8iZS8smByvJzxzVoATQsPAzpGRDgfqYQmOQJLf61abxvILkh1IQ1zSzaCx+fksai1oJ0JidxZ/o2dLIkIS4BToCVWV8wQJZc30GgY0DKuqat5LGvAoZyCjbnucuGouD4KNx2BqjKGRgyJ0FaSyeAp0UXcvQVZGxEuKAaTzGePVlMQDWwCbnml+dR+wHEkhsX3voIjnsG+n+IfsVYoSe5l+AfGtJY4ZemZIDEjVMYztiTEjjvAyP8/I1NX7JYTY+tTQCFYpN47wsM9NczUoas5KX1etu/KD/SjSgMfvC4BiBckhgeMUsLYbGdkxEH8rQO5NBY/WQtMyPRrfgJOthcwyxi0UmGirfIvYlvas6BSVVj3B0GuvRoIGWNOY4FuWbEpyExnD8Dj35dX6uRWPeBXnQZXLsFgeQnMqUNLYjdhJD6l9JY9bqrNp2P3MFxnkjGjNMcB6+yUiB+oI2Np7GMtKRl7eYUGusfyafC+N8603ZFjs8gYxs3EOToMayWRpOevN1DYqvPqSp05geptNEtKExZVdGrUnIujRm+fqJHBQ1EcpaNKZQGuE9PV7dSRb4aCw/scKVNM7V86JWeHwZpDDQgMZ00TOXXuD3/FwZG+tyqF+YW81LmM82URipMe1igur2OWNY256pKtZXa+qAUmhz1ZfYrIEZoLSgcGeazdFpTIIJ33S0oTGLFX+MFKpL41wJltje3LLPtWhMFcDoV7wPajHnQFGAxrwvmRZDYzw4vocX4BsKaUDjvHxA26/O1DHl/GwovjCN6bjJs+iz6wLz2SYNxdiVss+Kzx8+/Pw9xM9ionU+P/j0pLH9AIfmbcbaKVPhh61BOwWjQQSHNDt0cYVjjAskCETHU57a0Jgj3jJeQoFrQGNeSFcCFaOcSc5aNM7L8DjTaMZ6wkiKQZauMWiM2koefSbGoPFxNSNtBCkcK8o8Xk/elMYy+oA+Yq+xhnwNhp+ppyJprMfb5L+oA8Fk05jWVWxs+YGUhvlFSAz2yoLt8SYEGkiVzE9m/rrc/atzNtDBcnQ9+pjzCmIZZBZqcXFRXyZBkujeuJPG+lDnnAL/pXyzTMbJckf+KzRmfO4Ucd4iUxqBH/TTpA1FQxpLuMgK6DSmvjDCGYl1cSHEXepPGot2YyvhcUg5qQYbVhjOW2T7x4pkUsPX9WwQ0pjjpgaNfSEYW4CgI4BbkLr4F3XjFKDMk9k0Nk7nphpCU7DBbd062TT24dF42pn/oaGLwHj3w2Kfems/IDYfYbCX8E0P9AN0OZQ6eohxmm4xIOvwVPkTSEVpg+BCenR2ofvxjXQtaIN/hUVonHUF/V9IM2wX9AL+ikbYUEeiUWpzVTH863RfcyVtuBbiherhXnJVXtZHaoi74AFlxdJFdRakECxXPrnxRMQVddooyodMSx9tqRekRAvcVQuPTyjoHgoeCtl09wpty5aUqpLGzuEoAg2CZxQnCAWK+jAa2QZqiMtZsXW19D1Xz84Ww+OjhviehUM7oDR2PUvW1XSunA6UxmuNUXpiWmm8JzU0zuqD/qUONCLEyDiz74Q6NtaniQd0AZ3tBw56Mb6JiYmqaiWmk8ZNAdJeVxvQTFcnDUgOCAC3FpgNSXi8aDxgQK44fEINyzkHw6cRA40HPHbgmLZzPm9KMdB4wGOH7e1t31zxlGKg8YABU4+BxgMGTD0GGg8YMPUYaDxgwNRjoPGAAVOPgcYDBkw9BhoPGDD1GGg8YMDUY6DxgAFTj4HGAwZMPQYaDxgw9RhoPCAxLl++LLvnDOgHA40HpAQ32fHtNTmgIww0HpAMI7X9Mv5ub2/3tsfogHyg8eQA0r+6ukrpX19f920uM+E4evTo3Nwc/soWnwcIuPdsxp2dncXFxdnY6MOJyaKxbONaF/G6f0edXWh+q4BC4stJC9yXRyWcOHGC+yRmnl0pGwOl+fbHI2rt73tXwfy23OBK3/UuL/dO2svkQdqIWvavlT0VZUPZflBLmNFckY/vzLaPxrItI/rbmbtrLC0tOYXDCTw58ueKA888+1yk0KMvndsL5+qwAue2gz0AXc5NTPGXYvfE156MEf1IyJarzgNlctXjxnkuYZw6dcq5ewaMsC05eKJDT349vHvhhQsX7H1b24AKhTSGKc5cW8B3CmjkSGFmMGJsausE2taZbY/G7OnNzU3ulz0/Px9ZiVTgZsJhiyGQrZJhxEiAmGu523DmMnRU3ge4OROUEYwYZR3V0K1Ze8hGzfaD50qJwBOGrJs/eCBKwS7N5qpsB21vki5g41d2X11sKeSqDsaWsV2DkhapmGS78rD4ge0wV6ddm2OPacw9wQlhRbjQ5OBW2pExITqG2/BDxdJ8oS0qfRhuoe68C33aSYjobCYkARrKfmpCaGn+4IFoQ3ujZnwpW1ITd9UREJn/uFk598jnKbRHR00aAHdTjmSQuP1hRcZoy+lL7vUcT/ShfoU/0LP2yksi2ZLhA/pGJAbVruRwXtpw+1TrvGS4s416Btqh5xBdXJuYNsxdB1ARQkhoZN2b4MkJPtOEq3gAQEc0RrFywE1vOK0OxzA0mg90WALeChFF49wTPfcGHtTSdQBjnN0uoEZ0eix9ghFdTEdEUi4GQsv4MIqHrRj15JlsCINZmrjczOwL9UfqeJqsyhY1A3QE6OFzBLoDD0yLd+644W5YfTMyqqZxroY60Pq1nJCwzN1XML91gcqmAY1RYfRTuBoCnodkc4COpXP8oBIJpYQEiLFLtUakCEiVsy8kfI2nMZd5GMzknBmLIshkX5sLjHPnEoJnQTlH4zoFKRdPY+YPD4jwWZwiao5USwfESFKuaK8fdWtAgi7kqaQZaVzXmZdwN3I6wSl/eVnVBtaYp4H5PMa6oDNWaZfYU/ZTBECP16klhcZha6CDzeUkHvwa6DWZNoNo8hRFM58GRpLO0lqC47UBEc1V9I7Hgf0E2/G3svEJCCq8DJ+4kpbO1naCXlj41szjPHRqX+PyeGhBTKfSGfBpcTlxL1ML9JymQEA/pNbBaBzhROsz+gorM8I3Hs42aqC22Wic+moDGCs8Dg1XpVJgNt+QlRMyAWN3q9A4YDANkB6+CvAWtB6oKmisH2VogxNClU9dFzzxLKtS8ZRhOPZyypc9Am9A7JMvM61LZWwMnsOuQIQowGG9TGl3TijsozFHuTjUlEUwSqy380nyskCGRlkVSRgbRw7uEQjDaD8Z2drktMGctsTw+waeqvR9pbvhA8hz8Re/ZHxI+OghYLZazjyb11m4xMbG9wFwqtNniwQy0RVuWIqcXbHGgGSKGGeKomaOEjLqSRtAqoRrm5fRPuH0cmmTwuO1MkAtgMoLhLRkk1Ml7es5KAZKRqxpejQ+edXnPKAE+gDgGy1zQN/Qn4zvS85ksHxWOMYa++aHaTpqEYOgU525VEMk6HPyWeiLVipQSolTMfsgJtcetm1A41yJsnO4xQAnRX2KnmBEbXdKM4h1AXvRnhRRM1MJdJ+0PypApRyuLctHZ9ElkWPfddBHCzzRSB0fjxryRDgDaDRbBVDSnH7NvsfLNFkE8WI6KRB7UDggPeAYHpWVCDxYTH8b4KDoJXU8LO9l5rBAlWabEc6Fxo9J6KBurlVzgUz2cDCC/1Y6Y1RGTgEKgOMCtqqSOhjfh4H+iqkArWJ40IFaKSAbtUClIBaVsrE/iwndMFbqUKpaun7zCmaOsg4BmwQaF6HHo/G/KI1SJIdv20aeXHNOl+49Ho2bPAN6yO5vJxha2GEV4weqJRab+e12HvHkNijNRLXvoEBtYlejsTXOSwXUbIRGjHmmZg7pWlfSWGKzWi4AVZVdeANrDOaLI2Z3vYCKg3LvG0DJy0iyVtcHQIHMlDWW0RkzUwk8u1jjzDOdowNPwTiWSge3cI45xUeI19Qp7VIBgW0V2E0V1piZYN/Qmmx3uyAn2Ad4tpUSqBb7WFQLnrZS47KcZ559DoSUonw4rc6hpwvEu+CzmckCSqbOs2lMYqCe5jVV4GgT2i0gzWFAy0KJoH0Y1GTKzIYbga1KcAypEhIr2qGH+NvIY15mATpL/ECZIrZHJcBzZkOP0wsN+EoMoZvpQRu4EXoT4oT2REezx52PJoTPVLgnTYRoxcgJeUM7Gw7wiuoI2/vNS40Z7keRRgJdL9NDI4X9RY67yenX7NEYl+kDy4FRAQMyjOGDkDkL9qVuWruGPZ0mFqkZnH3ZALJ0sRJodmrGWrBdNUIX6GYwmtQYvwmP5FPobcFNAhnb8wGizuFJCS7CAJFokAmnl8HYLRJo/JjxUU4ZOg3hPmcDOpVOHZREYCxKB0cs0Q2yvFsHOo9eH4egKqOOyhY3AAsMdYsmoEscA2hc23QQslS4Lpq54j5UqsVMuyO91kjARfKpUZhorlGPAeQYbYWeRbPLVYZ7v7LfZQg7/3jkysHhxoAk6/bJgBEUQBnpFtIA2EEWoRkrJ8lilCwIHO+DkMZOreGIGex5xQCoxemn7ar3dSUZOfFNpLo1yvEl87KIC80LXLCvCqfIh6qHRxXVMLLbGZzJuMoGx13CybzmUTGJbY+ZQ9Dlksomgig7pTMh7AdxPEsJXzvoee4r6N84YRfiLC0GI7WywPxWwUHjSKDdGRY6Y+7HB84w5rECvKEYgR7QHWJpfF+9sQllAMMLp1SWQGZBv+JxAKKJlarhzQEDKgFjgFipmUKMpfHIWvVOpA0Lpw4SmtpD351CVCc+RI5iDJhwcJw8MH4RQCyNCUgM5JXTPPPz86mGZ6cXiCwWFhaeefa5rkM7HRwERuNzzhkdMTB5BsBhJoSondNYcL/OdmGzjTsK5redQZ+RknnOVAsnBhws4PA24HDemMYDDhBcTcGJCkTmldN4A2YeA42nEs109oBZxUDjAQOmHv8Py9l0nlTVoisAAAAASUVORK5CYII=>