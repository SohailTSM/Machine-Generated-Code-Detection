"""Central configuration — ENV toggle, task config, hyperparameters, paths."""

import os
import torch

# ═══════════════════════════════════════════════════════════════
# ██  USER-EDITABLE SETTINGS                                  ██
# ═══════════════════════════════════════════════════════════════
ENV = "kaggle"           # "kaggle" or "local"
ACTIVE_TASK = "A"        # "A", "B", or "C"
DEBUG = False            # True = small sample for fast iteration

HF_DATASET_NAME = "sohailtsm/my-semeval-dataset"  # HuggingFace dataset source
HF_MODEL_REPO = "dhruv10050/semeval-gnn-models"  # HuggingFace repo for model uploads
HF_WRITE_TOKEN = os.environ.get("HF_WRITE_TOKEN", "")
HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

# ═══════════════════════════════════════════════════════════════
# ██  PLATFORM CONFIG                                          ██
# ═══════════════════════════════════════════════════════════════
ENV_CONFIG = {
    "kaggle": {
        "work_dir":           "/kaggle/working",
        "data_dir":           "/kaggle/input",
        "device":             "cuda",
        "use_amp":            True,
        "num_workers":        2,
        "persistent_workers": True,
        "batch_size":         32,
        "codebert_batch":     768,
        "torch_compile":      False,
        "pin_memory":         True,
        "gradient_scaler":    True,
    },
    "local": {
        "work_dir":           ".",
        "data_dir":           ".",
        "device":             "mps",
        "use_amp":            True,
        "num_workers":        0,
        "persistent_workers": False,
        "batch_size":         16,
        "codebert_batch":     16,
        "torch_compile":      False,
        "pin_memory":         False,
        "gradient_scaler":    False,
    },
}

# ═══════════════════════════════════════════════════════════════
# ██  TASK CONFIG                                              ██
# ═══════════════════════════════════════════════════════════════
TASK_CONFIG = {
    "A": {
        "name":          "Binary Detection",
        "hf_dataset":    "DaniilOr/SemEval-2026-Task13",
        "hf_config":     "A",
        "num_classes":   2,
        "label_names":   ["human", "machine"],
        "label2idx":     {"human": 0, "machine": 1},
        "idx2label":     {0: "human", 1: "machine"},
        "focal_gamma":   2.0,
        "augment_ratio": 0.15,
        "train_label_remap": True,   # 11→2 remapping needed
        "max_train_samples": 100_000,
        "max_val_samples":   20_000,
    },
    "B": {
        "name":          "Authorship Attribution",
        "hf_dataset":    "DaniilOr/SemEval-2026-Task13",
        "hf_config":     "B",
        "num_classes":   11,
        "label_names":   ["Human", "DeepSeek-AI", "Qwen", "01-ai", "BigCode",
                          "Gemma", "Phi", "Meta-LLaMA", "IBM-Granite",
                          "Mistral", "OpenAI"],
        "label2idx":     {"Human": 0, "DeepSeek-AI": 1, "Qwen": 2, "01-ai": 3,
                          "BigCode": 4, "Gemma": 5, "Phi": 6, "Meta-LLaMA": 7,
                          "IBM-Granite": 8, "Mistral": 9, "OpenAI": 10},
        "idx2label":     {i: n for i, n in enumerate(
                          ["Human", "DeepSeek-AI", "Qwen", "01-ai", "BigCode",
                           "Gemma", "Phi", "Meta-LLaMA", "IBM-Granite",
                           "Mistral", "OpenAI"])},
        "focal_gamma":   3.0,
        "augment_ratio": 0.20,
        "train_label_remap": False,
        "max_train_samples": 100_000,
        "max_val_samples":   20_000,
    },
    "C": {
        "name":          "Hybrid Detection",
        "hf_dataset":    "DaniilOr/SemEval-2026-Task13",
        "hf_config":     "C",
        "num_classes":   4,
        "label_names":   ["human", "machine", "hybrid", "adversarial"],
        "label2idx":     {"human": 0, "machine": 1, "hybrid": 2, "adversarial": 3},
        "idx2label":     {0: "human", 1: "machine", 2: "hybrid", 3: "adversarial"},
        "focal_gamma":   2.5,
        "augment_ratio": 0.25,
        "train_label_remap": False,
        "max_train_samples": 100_000,
        "max_val_samples":   20_000,
    },
}

# ═══════════════════════════════════════════════════════════════
# ██  ARCHITECTURE HYPERPARAMETERS                             ██
# ═══════════════════════════════════════════════════════════════
HIDDEN_DIM = 768
EMB_DIM = 384
SEMANTIC_DIM = 384
EDGE_EMB_DIM = 96
EDGE_TYPES = 4
NUM_GNN_LAYERS = 5
TRANSFORMER_HEADS = 12
TFIDF_DIM = 100
VOCAB_SIZE = 8000
MAX_NODES = 512
MAX_UNIQUE_TEXTS = 500_000   # cap CodeBERT unique node texts (prevents hour-long embedding)
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1
CONT_DIM = 6

# ═══════════════════════════════════════════════════════════════
# ██  TRAINING HYPERPARAMETERS                                 ██
# ═══════════════════════════════════════════════════════════════
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 15
MAX_GRAD_NORM = 1.0
CHUNK_SIZE = 10000
GRADIENT_ACCUM_STEPS = 32   # effective batch = batch_size * accum_steps
DEBUG_TRAIN_SAMPLES = 500
DEBUG_VAL_SAMPLES = 100

# ═══════════════════════════════════════════════════════════════
# ██  CACHE VERSIONING                                         ██
# ═══════════════════════════════════════════════════════════════
GRAPH_LOGIC_VERSION = "v11_medium"

# ═══════════════════════════════════════════════════════════════
# ██  DERIVED GLOBALS (set by init_config)                     ██
# ═══════════════════════════════════════════════════════════════
BATCH_SIZE = None
WORK_DIR = None
DATA_DIR = None
CACHE_DIR = None
CHECKPOINT_DIR = None
SUBMISSION_DIR = None
LOG_DIR = None
DEVICE = None
USE_AMP = None
NUM_WORKERS = None
NUM_CLASSES = None
FOCAL_GAMMA = None
AUGMENT_RATIO = None
LABEL_NAMES = None
SEED = 42


def init_config():
    """Resolve all derived settings from ENV and ACTIVE_TASK. Call once at startup."""
    cfg = ENV_CONFIG[ENV]

    global BATCH_SIZE, WORK_DIR, DATA_DIR, CACHE_DIR, CHECKPOINT_DIR
    global SUBMISSION_DIR, LOG_DIR, DEVICE, USE_AMP, NUM_WORKERS
    global NUM_CLASSES, FOCAL_GAMMA, AUGMENT_RATIO, LABEL_NAMES

    BATCH_SIZE = cfg["batch_size"]
    WORK_DIR = cfg["work_dir"]
    DATA_DIR = cfg["data_dir"]
    CACHE_DIR = os.path.join(WORK_DIR, "pt_chunks")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submissions")
    LOG_DIR = os.path.join(WORK_DIR, "logs")

    for d in [CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # Device resolution
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif cfg["device"] == "mps" and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    USE_AMP = cfg["use_amp"]
    NUM_WORKERS = cfg["num_workers"]

    task_cfg = TASK_CONFIG[ACTIVE_TASK]
    NUM_CLASSES = task_cfg["num_classes"]
    FOCAL_GAMMA = task_cfg["focal_gamma"]
    AUGMENT_RATIO = task_cfg["augment_ratio"]
    LABEL_NAMES = task_cfg["label_names"]

    print(f"✓ Config: ENV={ENV} | TASK={ACTIVE_TASK} ({task_cfg['name']}) | "
          f"DEVICE={DEVICE} | BATCH={BATCH_SIZE} | DEBUG={DEBUG}")
