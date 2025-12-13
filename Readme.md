# 🚀 Zero to LLM: A Deep Learning Roadmap

**Instructor:** [Ukesh Thapa]  
**Repository Goal:** To build the intuition behind Large Language Models (LLMs) by replicating the historical evolution of NLP architectures.

> **Philosophy:** We do not start with the Transformer. We earn it. To understand *why* GPT-4 exists, we must first understand the failures of the systems that came before it.

---

## 📚 Syllabus Overview

| Phase | Era | Focus | Key Architecture | Tokenization | Evaluation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | 1950s–2013 | **Statistics** | Bag of Words, N-Grams | Stemming & Stop-words | Precision/Recall |
| **II** | 2013–2017 | **Sequences** | RNN, LSTM, Seq2Seq | Word-Level (Dict) | Perplexity, BLEU |
| **III** | 2014–2017 | **Alignment** | Attention Mechanism | Word-Level | Alignment Visuals |
| **IV** | 2017–2020 | **Parallelism** | Transformers (BERT) | Subword (WordPiece) | GLUE, ROUGE |
| **V** | 2020–Now | **Scale & Reasoning** | LLMs (GPT, Llama) | Byte-Level BPE | MMLU, HumanEval |

---

## 🛠 Prerequisites

* **Language:** Python 3.8+
* **Core Libraries:** `numpy` (Manual Math), `torch` (Auto-grad), `tiktoken` (Tokenization).
* **Math Foundation:**
    * Linear Algebra (Dot products, Matrix shapes).
    * Probability (Softmax, Log-Likelihood).
    * Calculus (Chain rule for Backpropagation).

---

## 🏛 Phase 1: The Statistical Era (1950s – 2013)
*Before "Brains," we used "Calculators."*

### 1.1 Bag of Words (BoW) & TF-IDF
**Concept:** Text is just a collection of frequencies. "Man bites dog" = "Dog bites man."
* **Tokenization:** **Linguistic Cleanup.**
    * *Steps:* Whitespace split $\rightarrow$ Lowercase $\rightarrow$ Remove stop-words ("the", "is") $\rightarrow$ Stemming ("running" $\to$ "run").
* **The Math:** Sparse Vectors (mostly zeros).
* **Metric:** **Precision/Recall/F1-Score.**
* **Assignment:** Build a Spam Filter from scratch using Python dictionaries.

### 1.2 N-Grams (The Probability Tables)
**Concept:** Predicting the next word based on the previous $N-1$ words.
* **The Math:** Markov Assumption.
    $$P(w_t | \text{history}) \approx P(w_t | w_{t-1})$$
* **The Flaw:** **Sparsity.** If the model hasn't seen the exact phrase "Purple Elephant" in training, the probability is 0.
* **Assignment:** Build a `Trigram` language model that generates nonsensical but grammatically strict text.

---

## 🧠 Phase 2: The Neural Era (2013 – 2017)
*Giving the model a Memory.*

### 2.1 Word Embeddings (Word2Vec / GloVe)
**Concept:** Words are no longer ID numbers; they are vectors in space. King - Man + Woman $\approx$ Queen.
* **Tokenization:** **Word-Level Dictionaries.**
    * Fixed Vocabulary (e.g., Top 50k words).
    * **The `<UNK>` Problem:** Any word not in the list becomes "Unknown."
* **Assignment:** Train your own embeddings on a small corpus and visualize them using PCA/t-SNE.

### 2.2 Recurrent Neural Networks (RNN & LSTM)
**Concept:** Processing data sequentially (Time step $t$). The "Conveyor Belt" of memory.
* **Architecture:**
    $$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$
    
* **The Flaw:** **Vanishing Gradient.** The signal dies over long sequences (50+ words).
* **Metric:** **Perplexity (PPL).** Measures how "surprised" the model is by real text. (Lower is better).
* **Assignment:** Build a Character-level RNN to generate text letter-by-letter.

---

## 🔦 Phase 3: The Bottleneck & The Solution (2014 – 2017)
*How we learned to translate.*

### 3.1 Seq2Seq (The Encoder-Decoder)
**Concept:** Compress English into *one* vector ($C$). Unpack it into French.
* **The Flaw:** **Information Bottleneck.** You cannot compress a book into a single vector.

### 3.2 The Attention Mechanism (Bahdanau/Luong)
**Concept:** The "Searchlight." The Decoder looks back at the Encoder's hidden states at every step.
* **The Math (Dot Product):**
    $$Score = \text{DecoderState} \cdot \text{EncoderState}^T$$
    *(Measures Similarity: Are these vectors pointing in the same direction?)*
* **Metric:** **BLEU Score.** Comparing overlap between machine translation and human references.
* **Assignment:** Implement `Attention` from scratch using NumPy. Plot the attention heatmap.

---

## ⚡ Phase 4: The Transformer Revolution (2017 – 2020)
*Attention Is All You Need.*

### 4.1 The Architecture
**Concept:** Kill the RNN. Process the whole sentence in parallel.
* **Key Components:**
    1.  **Positional Encodings:** Adding math waves ($sin/cos$) to words so the model knows the order.
    2.  **Self-Attention:** $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
    3.  **Multi-Head:** Thinking about Grammar, Vocabulary, and Tone simultaneously.
* **Tokenization:** **Subword (BPE/WordPiece).**
    * "Unhappiness" $\to$ `["Un", "happi", "ness"]`.
    * *Solves the `<UNK>` problem.*
* **Metric:** **GLUE Benchmark.** A suite of 9 difficult tasks (Logic, Paraphrasing, Sentiment).
* **Assignment:** Code a `SelfAttention` block (no `torch.nn`).
    

---

## 🤖 Phase 5: The Large Language Model (2020 – Present)
*The Era of GPT, Llama, and Mistral.*

### 5.1 Modern Decoder-Only Architecture
**Concept:** Stack massive amounts of Transformer Decoders. Train on the internet.
* **Upgrades from 2017:**
    * **RoPE (Rotary Embeddings):** Better handling of sequence length.
    * **SwiGLU:** Better activation function than ReLU.
    * **RMSNorm:** More stable training.
* **Tokenization:** **Byte-Level BPE (Tiktoken).**
    * Handles Emojis 🚀, Code, and foreign languages without `<UNK>`.
    * Tokens handle spaces: `_Hello`.

### 5.2 The Training Pipeline
1.  **Pre-Training:** Predicting the next token on Trillions of words.
2.  **SFT (Supervised Fine-Tuning):** Learning Q&A format ("Instruction Tuning").
3.  **RLHF (Reinforcement Learning):** Aligning with human preferences (Helpful, Honest, Harmless).

### 5.3 Modern Metrics (Vibes & Reasoning)
* **MMLU:** Massive Multitask Language Understanding (Math, Law, Medicine exams).
* **HumanEval:** Can the model write working Python code?
* **Chatbot Arena (Elo):** Human blind tests.

* **Final Assignment:** Build **"Mini-GPT"**.
    * Train a GPT-style model on the `TinyShakespeare` dataset.
    * Implement **Top-k Sampling** and **Temperature** for generation.

---

## 📜 Reading List (The Hall of Fame)

These are the seminal papers that defined each era.

| Year | Paper Title | Impact |
| :--- | :--- | :--- |
| **2003** | [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (Bengio et al.) | The first serious attempt at a Neural Network for Language. |
| **2013** | [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al.) | **Word2Vec**. Proved King - Man + Woman = Queen. |
| **2014** | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever et al.) | Defined the **Encoder-Decoder** bottleneck architecture. |
| **2014** | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al.) | Invented **Additive Attention** (The Searchlight). |
| **2015** | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al.) | Invented **BPE Tokenization** (Subwords). |
| **2017** | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al.) | Invented the **Transformer**. Killed RNNs. |
| **2018** | [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al.) | The peak of "Encoder-Only" models. |
| **2020** | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al.) | **GPT-3**. Proved that scale creates emergent reasoning. |
| **2022** | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al.) | **InstructGPT**. Introduced RLHF (Alignment). |
| **2023** | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al.) | Democratized powerful LLMs for the open-source community. |

---

## 🎓 Professor's Checklist for Students

Before moving to the next phase, ask yourself:
1.  **Phase 1:** Why does "Bag of Words" fail on the sentence "Not good, bad"?
2.  **Phase 2:** Why can't RNNs be trained on parallel GPUs easily?
3.  **Phase 3:** Why is the Dot Product a good measure of similarity?
4.  **Phase 4:** Why do we divide by $\sqrt{d_k}$ in Attention?
5.  **Phase 5:** Why does GPT only use the Decoder and not the Encoder?

---

*Happy Coding!*