# LoRA Fine-Tuning Architecture for Local Expense Categorization (Apple Silicon MLX + Ollama)

This document explains the technical concepts, parameter-efficient fine-tuning (PEFT) mechanics, data engineering pipeline, and local deployment workflow used to adapt a general-purpose Small Language Model (**Qwen2.5-Instruct**) into a domain-specific Indian household expense classifier without exposing private financial records to external APIs.

---

## 1. Why LoRA (Low-Rank Adaptation)?

Full fine-tuning of a 1.5B–3B parameter LLM requires updating all model weights (`W`), storing optimizer states (AdamW momentum + variance), and keeping gradients in memory—which demands 24–48+ GB of VRAM and risks catastrophic forgetting of the model's pre-trained world knowledge.

**LoRA** freezes the pre-trained base model weights (`W₀ ∈ ℝ^(d×k)`) and injects trainable low-rank decomposition matrices (`A` and `B`) into the transformer's attention and MLP projection layers:

```text
           Input Hidden State (x)
              │              │
              ▼              ▼
      ┌──────────────┐   ┌────────┐
      │  Frozen W₀   │   │  A     │  (r × d, initialized with Gaussian noise)
      │  (4-bit NF4) │   └───┬────┘
      └──────┬───────┘       ▼
             │           ┌────────┐
             │           │  B     │  (k × r, initialized to 0 -> ΔW = 0 at start)
             │           └───┬────┘
             │               │  (scaled by α / r)
             ▼               ▼
            (+) ◄────────────┘
             │
             ▼
     Output Hidden State: h = W₀x + (α / r) · BAx
```

### Key Technical Benefits
* **99%+ Parameter Reduction:** Instead of updating billions of weights, LoRA trains only ~0.5%–1.5% of parameters (`~10M–25M` trainable parameters stored in a ~20–40 MB `adapters.safetensors` file).
* **QLoRA Memory Efficiency:** Combining **4-bit quantized base weights (`W₀`)** with **16-bit (`bfloat16`/`float16`) LoRA adapters** allows a 1.5B–3B parameter model to fine-tune within **< 6 GB of Apple Unified Memory**.
* **Zero Inference Overhead After Fusion:** Once trained, `ΔW = (α / r) · BA` is mathematically added back into `W₀` (`W_fused = W₀ + ΔW`), producing a standard standalone model with **zero extra latency** at inference time.

---

## 2. End-to-End LoRA & Ollama Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION (Pandas + Regex Cleaning)                           │
│    • Strip UPI/SBI SMS boilerplate while keeping Unicode/Hindi (ऑटो)    │
│    • Normalize diagnostic/blood tests -> Medical                        │
│    • Upsample verified human corrections (data/corrections.csv) 2x      │
│    • Format as ChatML JSONL (train.jsonl / valid.jsonl)                 │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. PARAMETER-EFFICIENT FINE-TUNING (Apple Metal via mlx-lm)             │
│    • Base Model: Qwen2.5-1.5B / 3B-Instruct-4bit                        │
│    • Trainable Adapters on top 4–8 transformer layers (r=8, α=16)       │
│    • Gradient Checkpointing + Max Seq Length (192–256 tokens)           │
│    • Output: adapters/expense-lora/adapters.safetensors (~25 MB)        │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. ADAPTER FUSION & DEQUANTIZATION (mlx_lm fuse --dequantize)           │
│    • Merges W_fused = W₀ + (α/r)BA into standard HF safetensors         │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. QUANTIZATION & LOCAL SERVING (Ollama GGUF Q4_K_M)                    │
│    • Converts safetensors -> Q4_K_M GGUF (~1.1–1.9 GB)                  │
│    • Injects Qwen2.5 ChatML TEMPLATE + stop tokens (<|im_end|>)         │
│    • Enforces valid 23-category output via JSON Schema enum decoding    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Technical Breakdown

### Step 1: Prompt Engineering & ChatML Formatting
Each transaction is converted into a structured instruction-following triple (`system`, `user`, `assistant`) using Qwen2.5's native ChatML format:

```text
<|im_start|>system
Classify the Indian household expense into one valid category: Accessories, Appliances, Baby Care, Bills, Celebration, Education, Electronics, Entertainment, Fitness, Food Outside, Fuel, Furniture, Gift, Grocery, Grooming, House Help, Household, Maintenance, Medical, Milk, Other, Toys, Travel. Reply with ONLY the category name.<|im_end|>
<|im_start|>user
Head: Hotter recho | Comment: debited to VPA hotstaronline ybl JioHotstar | Amount: ₹149.00<|im_end|>
<|im_start|>assistant
Bills<|im_end|>
```

* **Loss Masking:** During LoRA training, cross-entropy loss is computed **only on the assistant response tokens** (`Bills<|im_end|>`), teaching the model to map transaction context directly to the exact category string without wasting capacity memorizing prompt boilerplate.
* **Active Learning Upsampling:** Human-verified edge cases from `data/corrections.csv` are duplicated (`2x`) in the training split so rare misclassifications receive higher gradient updates.

### Step 2: Apple Silicon Hardware Optimization (`mlx-lm`)
On a 16 GB Unified Memory MacBook Pro, memory pressure comes primarily from **KV-cache activations** and **intermediate gradients** rather than model weights:

| Hyperparameter | Value | Technical Rationale |
| :--- | :---: | :--- |
| `base_model` | `Qwen2.5-1.5B-Instruct-4bit` | ~900 MB base footprint leaves ample headroom for Metal GPU buffers without macOS swap thrashing. |
| `num_layers` | `4` (1.5B) / `8` (3B) | Adapts the top semantic transformer blocks where high-level classification decisions form, keeping lower lexical layers frozen. |
| `max_seq_length` | `192` – `256` | Expense prompts average ~95 tokens; capping sequence length cuts quadratic attention activation memory by ~75%. |
| `batch_size` | `2` | Prevents Metal command-buffer spikes on 16 GB unified memory. |
| `grad_checkpoint` | `True` | Recomputes activations during backpropagation instead of storing them in RAM, trading ~15% compute speed for ~50% lower peak memory. |
| `learning_rate` | `1e-4` – `1.5e-4` | Standard LoRA learning rate (roughly 10x higher than full fine-tuning because only low-rank adapters update). |

### Step 3: Adapter Fusion & Ollama GGUF Conversion
1. **Dequantized Fusion (`mlx_lm fuse --dequantize`):** MLX's 4-bit affine quantization format differs from `llama.cpp`/Ollama's `GGUF` block quantization. Dequantizing during fusion merges the LoRA weights cleanly into standard Hugging Face `safetensors`.
2. **ChatML Template Injection:** Ollama's `Modelfile` explicitly defines Qwen2.5's `<|im_start|>` / `<|im_end|>` delimiters and stop tokens so the model stops immediately after emitting the category name instead of continuing text generation.
3. **Grammar-Constrained Structured Output (`format: enum`):** When queried via `/api/chat`, passing a JSON Schema `enum` of the 23 valid categories masks logits for any invalid token at every decoding step—guaranteeing **0% out-of-vocabulary hallucinations**.

---

## 4. Empirical Evaluation: LoRA SLM vs. Classical ML Pipeline

Evaluated on the same stratified holdout split (`1,125` unseen Indian household transactions):

| Metric | Classical Pipeline (`TF-IDF + LogisticRegression`) | LoRA Fine-Tuned SLM (`Qwen2.5-4bit`) |
| :--- | :---: | :---: |
| **Overall Holdout Accuracy** | **96.00%** | **92.71%** |
| **Macro F1-Score** | **0.92** | **0.85** |
| **Out-of-Vocabulary Hallucinations** | 0% (fixed class array) | **0%** (with ChatML + `enum` grammar) |
| **Single-Row Latency** | **~0.5 ms** (CPU) | **~30–80 ms** (Apple Metal GPU) |
| **Batch Latency (2,500 rows)** | **~50 ms** (gRPC stream) | **~2–3 minutes** |
| **Runtime Memory Footprint** | **~80 MB RAM** (~5 MB artifact) | **~1.5–2.5 GB Unified Memory** |

### Why Each Model Excels at Different Edge Cases
* **Where Classical ML (`TF-IDF + LogisticRegression`) Wins:**
  * **Continuous Numeric Thresholds:** `StandardScaler(log_amount)` cleanly separates `Dishwasher` at `₹459` (`Grocery` detergent) from `₹32,472` (`Appliances`), whereas LLMs tokenize digits as text strings (`4`, `5`, `9`) and rely heavily on word semantics.
  * **High-Throughput Microservices:** Runs inside a minimal Python Docker container with zero GPU requirements.
* **Where the LoRA SLM (`Qwen2.5`) Wins:**
  * **Unseen Brand & Semantic Generalization:** Leverages pre-trained knowledge of Indian commerce (`JioHotstar` → `Bills`, `COMPASS INDIA FOOD SERVICES` → `Food Outside`, `Thyrocare` → `Medical`, Hindi `ऑटो` → `Travel`) even when exact tokens were rare in historical logs.
  * **Offline Active-Learning Assistant:** Ideal for pre-labeling ambiguous transactions in `data/corrections_pending.csv` on a developer laptop before human approval.