"""Shared utilities — memory monitoring, seeding, WandB/HF helpers."""

import gc
import os
import random
import time

import numpy as np
import torch

from src import config as cfg


def seed_everything(seed: int = None):
    """Set all random seeds for reproducibility."""
    seed = seed or cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mem_report(tag: str = ""):
    """Print current RAM + GPU memory usage."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / 1e9
        msg = f"[MEM {tag}] RAM: {rss:.2f} GB"
    except ImportError:
        msg = f"[MEM {tag}] (psutil not installed)"

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resrv = torch.cuda.memory_reserved() / 1e9
        msg += f" | GPU alloc: {alloc:.2f} GB, reserved: {resrv:.2f} GB"
    print(msg)


def disk_report(path: str = None, tag: str = ""):
    """Print disk usage of working directory."""
    import shutil
    path = path or cfg.WORK_DIR
    try:
        total, used, free = shutil.disk_usage(path)
        # Also measure working directory size
        work_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    work_size += os.path.getsize(fp)
                except OSError:
                    pass
        print(f"[DISK {tag}] Working dir: {work_size / 1e9:.2f} GB "
              f"| Disk free: {free / 1e9:.1f} GB / {total / 1e9:.1f} GB")
    except Exception as e:
        print(f"[DISK {tag}] Error: {e}")


def cleanup(*objs):
    """Delete objects and run garbage collection."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_wandb(run_name: str = None) -> bool:
    """Initialize WandB if API key is set. Returns True if active."""
    if not cfg.WANDB_API_KEY:
        return False
    try:
        import wandb
        wandb.login(key=cfg.WANDB_API_KEY)
        wandb.init(
            project="semeval2026-task13",
            name=run_name or f"v7_task{cfg.ACTIVE_TASK}_{cfg.ENV}",
            config={
                "env": cfg.ENV,
                "task": cfg.ACTIVE_TASK,
                "num_classes": cfg.NUM_CLASSES,
                "hidden_dim": cfg.HIDDEN_DIM,
                "num_layers": cfg.NUM_GNN_LAYERS,
                "heads": cfg.TRANSFORMER_HEADS,
                "lr": cfg.LEARNING_RATE,
                "batch_size": cfg.BATCH_SIZE,
                "dropout": cfg.DROPOUT,
                "focal_gamma": cfg.FOCAL_GAMMA,
                "augment_ratio": cfg.AUGMENT_RATIO,
                "label_smoothing": cfg.LABEL_SMOOTHING,
                "max_nodes": cfg.MAX_NODES,
                "vocab_size": cfg.VOCAB_SIZE,
                "graph_logic_version": cfg.GRAPH_LOGIC_VERSION,
            },
        )
        print("✓ WandB initialized")
        return True
    except Exception as e:
        print(f"⚠ WandB init failed: {e}")
        return False


def init_hf():
    """Login to HuggingFace Hub if tokens are set."""
    token = cfg.HF_WRITE_TOKEN or cfg.HF_READ_TOKEN
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token)
        print("✓ HuggingFace login OK")
    except Exception as e:
        print(f"⚠ HuggingFace login failed: {e}")


class Timer:
    """Simple context manager for timing blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"  ⏱ {self.label}: {self.elapsed:.1f}s")
