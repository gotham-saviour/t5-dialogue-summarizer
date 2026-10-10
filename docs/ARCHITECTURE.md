# Deep Learning & NLP Architecture: Abstractive Dialogue Summarization

## 1. Executive Summary & Problem Formulation

Text summarization is broadly partitioned into two distinct paradigms:
1. **Extractive Summarization:** Heuristically or statistically extracting salience-ranked sentences directly from the source text without linguistic transformation.
2. **Abstractive Summarization:** Semantically interpreting the input context and synthesizing a novel, grammatically coherent target sequence that paraphrases core themes.

### Why Dialogue Summarization is a Hard Problem
Document summarization models trained on structured corpora (e.g., CNN/DailyMail, BBC, or Wikipedia) fail catastrophically when applied to conversational chat transcripts due to intrinsic structural divergence:
- **Discontinuous Turn-Taking:** Unlike written paragraphs with single-author topical cohesion, dialogue consists of alternating speaker turns across multiple participants.
- **High Coreference Ambiguity:** Pronouns (*"he said that to her"*, *"they agreed on it"*) reference entities established several turns prior.
- **Informal Syntax & Noise:** Chat transcripts exhibit slang, phonetic spellings, contractions, ellipses, and unpunctuated sentence fragments.
- **Implicit Speech Acts:** Intentions (commitments, disagreements, questions, schedule confirmations) are expressed implicitly through dialogue pairs rather than explicit declarations.

This project fine-tunes **Google's T5 (Text-to-Text Transfer Transformer)** on the **SAMSum corpus** to formulate abstractive dialogue summarization as a conditional sequence-to-sequence language generation task:

$$\mathcal{Y}^* = \arg\max_{\mathcal{Y}} P(\mathcal{Y} \mid \mathcal{X}; \theta)$$

where $\mathcal{X} = \{x_1, x_2, \dots, x_M\}$ represents the multi-speaker conversational dialogue and $\mathcal{Y} = \{y_1, y_2, \dots, y_N\}$ represents the concise narrative summary.

---

## 2. Transformer Architecture: Google T5

T5 (Raffel et al., 2020) unifies every natural language problem—from translation and question answering to classification and summarization—into a uniform **text-to-text** formulation where inputs and outputs are token sequences.

```mermaid
graph TD
    subgraph Input Processing
        Raw["Input Multi-Turn Dialogue"] --> Clean["Preprocessing (HTML strip, whitespace normalization)"]
        Clean --> Tok["SentencePiece Tokenizer (Vocab: 32,128)"]
        Tok --> Embed["Token Embeddings + Relative Position Bias"]
    end

    subgraph Encoder Stack (6 Layers)
        Embed --> EncAttn["Multi-Head Self-Attention (8 Heads)"]
        EncAttn --> EncNorm1["LayerNorm (No Bias)"]
        EncNorm1 --> EncFFN["Feed-Forward Dense (d_ff = 2048)"]
        EncFFN --> EncNorm2["LayerNorm (No Bias)"]
        EncNorm2 --> EncOut["Contextual Hidden Representations H"]
    end

    subgraph Decoder Stack (6 Layers)
        ShiftTargets["Shifted Target Tokens (Teacher Forcing)"] --> DecAttn["Causal Masked Multi-Head Self-Attention"]
        DecAttn --> DecNorm1["LayerNorm"]
        DecNorm1 --> CrossAttn["Multi-Head Cross-Attention (Keys/Values from H)"]
        EncOut -.-> CrossAttn
        CrossAttn --> DecNorm2["LayerNorm"]
        DecNorm2 --> DecFFN["Feed-Forward Dense (d_ff = 2048)"]
        DecFFN --> DecNorm3["LayerNorm"]
        DecNorm3 --> LMHead["Linear Projection to Vocabulary Logits"]
    end

    subgraph Output Generation
        LMHead --> Beam["Beam Search Decoding (k=4, no_repeat_ngram=3)"]
        Beam --> Summary["Final Abstractive Summary"]
    end
```

### 2.1 Multi-Head Self-Attention with Relative Position Bias
Standard vanilla Transformers (Vaswani et al., 2017) inject absolute positional coordinates via sinusoidal wave functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

However, absolute positional coordinates do not generalize well across variable sequence lengths. T5 replaces absolute embeddings with **learned relative position bias**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B_{\text{rel}}\right)V$$

where:
- $Q = X W_Q$, $K = X W_K$, $V = X W_V$ with projection matrices $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $B_{\text{rel}} \in \mathbb{R}^{L \times L}$ represents a bucketed relative position bias matrix. For tokens at positions $i$ and $j$, the scalar bias corresponds to relative displacement $\Delta = j - i$, sharing parameters across large offsets logarithmically to maintain computational tractability across 128 distance buckets.

### 2.2 Simplified Layer Normalization
Unlike standard LayerNorm which learns both an affine scale $\gamma$ and shift bias $\beta$:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

T5 omits mean centering and bias shifting, scaling solely by the root-mean-square (RMS) norm:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

This reduces parameter count and improves gradient propagation stability through deep layers.

---

## 3. Sequence-to-Sequence Tokenization Pipeline

### 3.1 SentencePiece Subword Tokenization
T5 relies on a unigram SentencePiece subword vocabulary with **32,128 tokens**. SentencePiece treats the input string as a raw byte/character stream, escaping whitespace with a meta-symbol (`_`).

Benefits for conversational NLP:
- **Out-of-Vocabulary (OOV) Immunity:** Any unseen colloquialism, username, or slang term is broken down into subword morphemes without triggering `<unk>` tokens.
- **Language Agnostic Segmentation:** Whitespace is not treated as a hard boundary, ensuring robustness to typos and missing punctuation.

### 3.2 Tensor Shapes and Masking
During batching, inputs and targets are aligned as:
- **Input Tensor (`input_ids`):** Shape `[B, 512]` containing token indices.
- **Attention Mask (`attention_mask`):** Shape `[B, 512]` with binary indicators ($1$ for valid contextual tokens, $0$ for padding tokens).
- **Target Labels (`labels`):** Shape `[B, 140]` representing decoder ground truth. Cross-entropy loss computation ignores padding tokens via loss masking.

---

## 4. Decoding Strategies: Beam Search Optimization

Autoregressive decoders output a categorical probability distribution over vocabulary $V$ at each step $t$:

$$P(y_t \mid y_{<t}, \mathcal{X}) = \text{softmax}\left(W_{\text{vocab}} h_t^{\text{dec}}\right)$$

### 4.1 Greedy Search vs Beam Search
- **Greedy Search:** Chooses $\hat{y}_t = \arg\max_{w \in V} P(w \mid y_{<t}, \mathcal{X})$. Prone to sub-optimal local traps because an early high-probability choice can lead to low-probability completions later.
- **Beam Search ($k=4$):** Maintains a beam of the top-$k$ most probable partial sequence hypotheses. At each step, all $k \times |V|$ extensions are evaluated and pruned to keep the top $k$:

$$\mathcal{H}_t = \text{Top-}k \left( \left\{ (\mathbf{y}, w) : \mathbf{y} \in \mathcal{H}_{t-1}, w \in V \right\}, \text{score}(\mathbf{y}) + \log P(w \mid \mathbf{y}, \mathcal{X}) \right)$$

### 4.2 Degenerative Repetition Mitigation
Abstractive neural models frequently suffer from cyclic n-gram repetition loops (e.g., *"they went to the store and they went to the store"*). We enforce a hard constraint during generation:

$$\text{no\_repeat\_ngram\_size} = 3$$

If a candidate 3-gram matches any previously generated 3-gram in the current hypothesis, its generation logit is set to $-\infty$, preventing repetitive cycles.

---

## 5. Production Microservice Architecture

The inference pipeline is served via an asynchronous **FastAPI** web application engineered with:
1. **Dynamic Artifact Resolution:** Intelligently discovers fine-tuned checkpoint weights across directory levels, falling back to upstream base models if run in fresh environments without weights.
2. **Hardware Acceleration Auto-Detection:** Dynamically routes tensor execution to `NVIDIA CUDA` if available, `Apple MPS` on macOS, or multi-threaded `CPU` execution.
3. **Pydantic Data Contracts:** Enforces input schema validation (character limits, whitespace stripping) and formats structured responses with latency benchmarks.
4. **Interactive Jinja2 Dashboard:** Serves a responsive web UI equipped with 1-click dialogue presets, character counters, execution latency badges, and clipboard integrations.

