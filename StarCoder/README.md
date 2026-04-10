# SemEval 2026 Task 13 — Human vs AI Code Detection using StarCoder2 - 3B

Submission for **SemEval 2026 Task 13: Distinguishing AI-Generated Code from Human-Written Code**, covering all three subtasks. The system uses a hybrid architecture combining a frozen StarCoder2-3B backbone with handcrafted stylometric, pattern, and AST features.

---

## Tasks

| Subtask | Problem | Classes | Metric |
|---|---|---|---|
| **A** | Binary AI vs. Human detection | 2 | Macro F1 |
| **B** | LLM family authorship attribution | 11 | Macro F1 |
| **C** | Hybrid code detection | 4 | Macro F1 |

### Task A — Binary Detection
Classify code as **Human-written** or **AI-generated**. Eight programming languages, 30+ generator families.

### Task B — Authorship Attribution
Identify which LLM family produced a code snippet: Human, DeepSeek, Qwen, 01-AI, BigCode, Gemma, Phi, Meta-LLaMA, IBM Granite, Mistral, or OpenAI.

### Task C — Hybrid Code Detection
Classify code as one of:
- **Human-written** — entirely authored by a human
- **Machine-generated** — produced entirely by an LLM
- **Hybrid** — partially written or completed by an LLM
- **Adversarial** — generated via adversarial prompts or RLHF to mimic human style

### Final Data and Trained Models
- [HuggingFace Model URL](https://huggingface.co/Kathan0/semeval2026-task13-models/tree/main)
- [HuggingFace Repo (Data used for competition results)](https://huggingface.co/datasets/Kathan0/INLP_Endsem/tree/main)

---

## Results

| Subtask | Baseline | Our Score | Improvement |
|---|---|---|---|
| A | 0.30530 | **0.8938** | +192.76% increase |
| B | 0.2286 | **0.2900** | +26.9% relative |
| C | 0.4812 | **0.5559** | +15.5% relative |

---

## Architecture

All three subtasks share the same hybrid classifier:

```
Code Snippet
    │
    ├─► StarCoder2-3B (frozen, 8-bit) ──► mean pool ──► LayerNorm ──► [3072]
    │                                                                       │
    └─► Handcrafted Features (102-dim) ──► MLP ─────────────────────► [128]
                                                                            │
                                                              concat [3200] ──► Fusion MLP ──► logits
```

**Backbone:** `bigcode/starcoder2-3b` — frozen throughout training, loaded with 8-bit quantization via BitsAndBytes. Embeddings are pre-extracted and cached as `.pt` files.

**Handcrafted features (102-dim):**
| Group | Dim | Description |
|---|---|---|
| Stylometric | 12 | Line length, indent style, blank line ratio, comment density |
| Pattern | 57 | Regex counts for keywords, operators, naming conventions |
| AST | 33 | Node type distributions (functions, loops, conditionals) |

**Fusion head (~8M trainable params):**
- Feature MLP: `102 → 256 → 128` with LayerNorm + GELU + Dropout
- Fusion MLP: `3200 → 512 → 256 → num_classes` with LayerNorm + GELU + Dropout
- Loss: CrossEntropyLoss with inverse-frequency class weights

---

## Repository Structure

```
INLP_Endsem/
├── TaskA/
│   ├── config.yaml              # Training and data configuration
│   ├── model.py                 # Hybrid classifier (TaskAModel)
│   ├── dataset.py               # CachedDataset — loads pre-extracted .pt files
│   ├── extract_features.py      # Step 1: CPU — extract 102-dim handcrafted features
│   ├── extract_embeddings.py    # Step 2: GPU — extract 3072-dim StarCoder2 embeddings
│   ├── train.py                 # Step 3: Train fusion head
│   ├── evaluate.py              # Step 4: Evaluate on validation set
│   ├── generate_submission.py   # Step 5: Generate competition submission CSV
│   ├── run_pipeline.py          # Run all steps in order
│   └── REPORT.md                # Methodology and results report
├── TaskB/
│   ├── balance_data.py          # Balance raw 500k dataset by (language, label)
│   ├── config.yaml
│   ├── model.py                 # TaskBModel — 11-class output
│   ├── dataset.py
│   ├── extract_features.py
│   ├── extract_embeddings.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_submission.py
│   ├── run_pipeline.py
│   └── REPORT.md
├── TaskC/
│   ├── balance_data.py          # Balance raw 900k dataset by (language, label)
│   ├── config.yaml
│   ├── model.py                 # TaskCModel — 4-class output
│   ├── dataset.py
│   ├── extract_features.py
│   ├── extract_embeddings.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_submission.py
│   ├── run_pipeline.py
│   └── REPORT.md
├── features/                    # Shared handcrafted feature extractors
│   ├── stylometric.py           # 12-dim style features (indentation, naming, comments)
│   ├── pattern_detector.py      # 57-dim LLM-signature pattern features
│   ├── ast_extractor.py         # 33-dim AST structural features
│   └── __init__.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended; 16GB for batch extraction)
- [Competition datasets](https://github.com/mbzuai-nlp/SemEval-2026-Task13)

### Installation

```bash
git clone https://github.com/Kathan0/INLP_Endsem.git
cd INLP_Endsem
pip install -r requirements.txt
```

### Dataset Layout
Place the downloaded competition data in sibling directories of `INLP_Endsem/`:

```
sem-eval-2026-task-13-subtask-a/Task_A/  ← train.parquet, validation.parquet, test.parquet
sem-eval-2026-task-13-subtask-b/Task_B/
sem-eval-2026-task-13-subtask-c/Task_C/
```
Or download data from huggingface repo.

Update the paths in each `TaskX/config.yaml` if needed.

---

## Usage

Each subtask follows the same 5-step pipeline. You can run all steps at once or individually.

### Full Pipeline (any task)

```bash
cd TaskA   # or TaskB / TaskC
python run_pipeline.py
```

### Step-by-Step

```bash
cd TaskA

# Step 1 — Extract handcrafted features (CPU, ~30 min for 50k samples)
python extract_features.py --splits train val test

# Step 2 — Extract StarCoder2 embeddings (GPU, reduce batch_size if OOM)
python extract_embeddings.py --splits train val test --batch_size 16

# Step 3 — Train the fusion head (~8M params)
python train.py

# Step 4 — Evaluate on validation set
python evaluate.py

# Step 5 — Generate competition submission
python generate_submission.py
```

### Task B and C — Balance Raw Data First

```bash
cd TaskB
python balance_data.py   # creates TaskB/data/ from raw 500k
# then follow the 5-step pipeline above

cd TaskC
python balance_data.py   # creates TaskC/data/ from raw 900k
```

### Resume Training

```bash
python train.py                   # auto-resumes from latest checkpoint
python train.py --no-auto-resume  # start from scratch
python train.py --resume checkpoints/epoch_10.pt  # resume from specific epoch
```

### Pipeline Flags

```bash
python run_pipeline.py --start-from 3       # skip extraction, start from training
python run_pipeline.py --steps 1 2          # run only extraction steps
python run_pipeline.py --no-auto-resume     # train from scratch
```

---

## Key Design Decisions

**Pre-extraction caching** — The StarCoder2-3B backbone is frozen, so embeddings are computed once and saved as `.pt` files. This makes each training epoch fast (only the 8M-param head is updated) without re-running the 3B model.

**Per-(language, label) balancing** — Raw datasets are heavily skewed (e.g., Task B: 88% human). Instead of discarding all majority data, each (language, label) group is capped at a fixed count, preserving minority combinations in full.

**8-bit quantization** — StarCoder2-3B is loaded with `BitsAndBytesConfig(load_in_8bit=True)`, reducing GPU memory from ~12GB to ~4GB with negligible accuracy loss.

**Inverse-frequency class weights** — CrossEntropyLoss is weighted by the inverse frequency of each class in the training set, mean-normalised so the average weight equals 1.

---

## Reports

Detailed methodology and analysis for each subtask:

- [TaskA/REPORT.md](TaskA/REPORT.md)
- [TaskB/REPORT.md](TaskB/REPORT.md)
- [TaskC/REPORT.md](TaskC/REPORT.md)
