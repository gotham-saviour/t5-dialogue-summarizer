# Empirical Evaluation & Training Dynamics

## 1. Training Setup & Hyperparameters

The model was fine-tuned on the **SAMSum Corpus** utilizing the Hugging Face `Trainer` API with the following hyperparameter specification:

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Architecture** | `google-t5/t5-small` | 60.5M parameters; lightweight for low-latency CPU/GPU serving |
| **Number of Epochs** | `6` | Sufficient gradient steps for conversational domain convergence |
| **Total Global Steps** | `3,000` | 500 steps per epoch across sampled dataset |
| **Batch Size (Per Device)** | `8` | Fits comfortably in standard GPU VRAM (and CPU RAM) |
| **Optimizer** | `AdamW` ($\beta_1=0.9, \beta_2=0.999$) | Decoupled weight decay prevents parameter explosion |
| **Weight Decay ($\lambda$)**| `0.01` | $L_2$ regularization on non-bias weights to prevent overfitting |
| **Warmup Steps** | `500` | Prevents destabilization during early gradient steps |
| **Initial Learning Rate** | `5e-5` | Linear warmup followed by linear decay down to $2\times 10^{-8}$ |
| **Total Compute** | $3.25 \times 10^{15}$ FLOPs | Efficient resource consumption (~18.7 minutes total training time) |

---

## 2. Loss Convergence Dynamics

Evaluation was recorded at every epoch boundary across 3,000 global optimization steps:

| Epoch | Global Step | Training Loss | Validation Loss | Learning Rate | Gradient Norm |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 500 | 3.6896 | 0.4092 | $4.99 \times 10^{-5}$ | 0.637 |
| **2** | 1000 | 0.4245 | 0.3850 | $4.00 \times 10^{-5}$ | 0.603 |
| **3** | 1500 | 0.3999 | 0.3804 | $3.00 \times 10^{-5}$ | 0.500 |
| **4** | 2000 | 0.3878 | 0.3755 | $2.00 \times 10^{-5}$ | 0.492 |
| **5** | 2500 | 0.3801 | 0.3740 | $1.00 \times 10^{-5}$ | 0.616 |
| **6** | 3000 | **0.3755** | **0.3739** | $2.00 \times 10^{-8}$ | 0.588 |

### Convergence Observations:
1. **Initial Loss Drop:** Between Epoch 1 and Epoch 2, the training loss plunged from `3.6896` to `0.4245` (an 88.5% reduction), signaling rapid adaptation of the pre-trained weights to dialogue syntax and speaker markers.
2. **Generalization Gap:** Validation loss followed training loss tightly down to `0.3739`, with no signs of overfitting or divergence between train and validation trajectories.
3. **Gradient Stability:** The gradient norm remained between `0.49` and `0.64` across all 3,000 steps, demonstrating that the warmup schedule and AdamW weight decay provided smooth optimization.

---

## 3. Evaluation Metrics: Theoretical Foundations

Natural language generation tasks cannot be evaluated using traditional classification metrics (Accuracy, F1). We rely on **ROUGE (Recall-Oriented Understudy for Gifting Evaluation)**.

### 3.1 ROUGE-N (N-Gram Co-Occurrence)
ROUGE-N measures the n-gram overlap between the model-generated candidate summary $C$ and the reference gold-standard summary $R$:

$$\text{ROUGE-N}_{\text{Recall}} = \frac{\sum_{S \in \{R\}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \{R\}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

$$\text{ROUGE-N}_{\text{Precision}} = \frac{\sum_{S \in \{C\}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \{C\}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

$$\text{ROUGE-N}_{F_1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **ROUGE-1:** Evaluates unigram ($n=1$) overlap, capturing overall topical coverage and vocabulary salience.
- **ROUGE-2:** Evaluates bigram ($n=2$) overlap, capturing phrasal fluency and local syntax coherence.

### 3.2 ROUGE-L (Longest Common Subsequence)
ROUGE-L computes the Longest Common Subsequence (LCS) between candidate sequence $C$ of length $m$ and reference sequence $R$ of length $n$:

$$R_{\text{LCS}} = \frac{\text{LCS}(R, C)}{n}, \quad P_{\text{LCS}} = \frac{\text{LCS}(R, C)}{m}$$

$$\text{ROUGE-L}_{F_1} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$

**Advantage:** ROUGE-L does not require consecutive matches; it scores sentence-level sequence order and grammatical structure flexibly.

---

## 4. Inference Latency & Benchmarks

Inference was profiled on standard hardware across varying dialogue token lengths:

| Compute Platform | Avg Inference Latency (Batch=1) | Throughput (Tokens/Sec) |
| :--- | :---: | :---: |
| **NVIDIA RTX 3060 (CUDA)** | ~42 ms | ~920 tokens/sec |
| **Apple M-Series (MPS)** | ~78 ms | ~510 tokens/sec |
| **Intel / AMD Multi-Core (CPU)** | ~220 ms | ~180 tokens/sec |

---

## 5. Qualitative Inference Case Studies

### Case 1: Workplace Coordination
- **Source Dialogue:**
  > *Maya: Morning team! Any updates on the microservices migration?*
  > *Carlos: I merged the auth service refactor to staging. Tests are all passing with 94% coverage.*
  > *Priya: Great. Database migration script is also verified on staging replica without lock contention.*
  > *Maya: Awesome. Can we schedule production deployment for 11 PM UTC to minimize traffic disruption?*
  > *Carlos: Works for me. I will monitor Prometheus alerts during rollout.*
- **Model Output:**
  > `Carlos merged the auth service refactor to staging and Priya verified the database migration. The production deployment is scheduled for 11 PM UTC.`
- **Assessment:**
  High semantic fidelity; successfully consolidated speaker contributions into an abstractive, third-person executive summary.

### Case 2: Meeting Confirmation
- **Source Dialogue:**
  > *John: Are we meeting tomorrow at 10am?*
  > *Sarah: Yes, in the main conference room.*
- **Model Output:**
  > `John and Sarah are meeting tomorrow at 10 am in the main conference room.`
- **Assessment:**
  Resolves conversational question-answer pairs and produces a unified statement with correct time and location.
