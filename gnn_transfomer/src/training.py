"""Training loop — train, evaluate, and orchestrate model training."""

import time
import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score

from src import config as cfg
from src.losses import FocalLoss
from src.utils import mem_report


# ─── Helpers ───────────────────────────────────────────────────
def _amp_dtype(device):
    """Return autocast dtype for the current device."""
    if device.type == "mps":
        return torch.bfloat16
    return torch.float16


# ─── Train one epoch ──────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                device, use_amp, accum_steps=1):
    """Run one training epoch with gradient accumulation.

    Args:
        accum_steps: Number of mini-batches to accumulate before optimizer step.
                     Effective batch = loader.batch_size * accum_steps.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.autocast(device_type=device.type,
                            dtype=_amp_dtype(device),
                            enabled=use_amp):
            logits = model(
                batch.x_type, batch.x_cont, batch.x_semantic,
                batch.edge_index, batch.edge_type,
                batch.batch, batch.tfidf,
            )
            loss = criterion(logits, batch.y) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step every accum_steps mini-batches (or at end of epoch)
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()

            scheduler.step()  # OneCycleLR steps per optimizer step
            optimizer.zero_grad()

            # Free cached GPU memory after each optimizer step to avoid fragmentation
            if device.type == "cuda":
                torch.cuda.empty_cache()

        total_loss += loss.item() * accum_steps * batch.y.size(0)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch.y.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1


# ─── Evaluate one epoch ──────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion, device, use_amp):
    """Run one evaluation epoch. Returns dict with loss, f1, bal_acc, preds, labels."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type,
                            dtype=_amp_dtype(device),
                            enabled=use_amp):
            logits = model(
                batch.x_type, batch.x_cont, batch.x_semantic,
                batch.edge_index, batch.edge_type,
                batch.batch, batch.tfidf,
            )
            loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.y.size(0)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch.y.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "f1": f1,
        "bal_acc": bal_acc,
        "preds": all_preds,
        "labels": all_labels,
    }


# ─── Full training loop ──────────────────────────────────────
def train_model(model, train_loader, val_loader, class_weights, device,
                resume_path=None):
    """Complete training loop with early stopping, AMP, WandB logging, and resume support.

    Args:
        resume_path: Path to a resume checkpoint to continue training from.

    Returns:
        history: dict with per-epoch metrics (train_loss, val_loss, etc.)
    """
    # Criterion
    criterion = FocalLoss(
        gamma=cfg.FOCAL_GAMMA,
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=cfg.LABEL_SMOOTHING,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    # Scheduler — steps_per_epoch counts optimizer steps (not mini-batches)
    accum_steps = getattr(cfg, 'GRADIENT_ACCUM_STEPS', 1)
    # ceil division: accounts for leftover batch at epoch end
    steps_per_epoch = max(1, -(-len(train_loader) // accum_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.LEARNING_RATE,
        epochs=cfg.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )

    # AMP scaler (CUDA-only)
    use_scaler = (device.type == "cuda") and cfg.USE_AMP
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # Checkpoint paths
    best_ckpt_path = os.path.join(
        cfg.CHECKPOINT_DIR,
        f"best_model_task{cfg.ACTIVE_TASK}.pt")
    resume_ckpt_path = os.path.join(
        cfg.CHECKPOINT_DIR,
        f"resume_task{cfg.ACTIVE_TASK}.pt")

    # Tracking
    start_epoch = 1
    best_f1 = 0.0
    patience_ctr = 0
    history = {
        "train_loss": [], "train_f1": [],
        "val_loss": [], "val_f1": [], "val_bal_acc": [], "lr": [],
    }

    # ── Resume from checkpoint ────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        patience_ctr = ckpt["patience_ctr"]
        history = ckpt["history"]
        print(f"\n✓ Resumed from epoch {ckpt['epoch']} "
              f"(best F1={best_f1:.4f}, patience={patience_ctr}/{cfg.PATIENCE})")
    elif resume_path:
        print(f"\n⚠ Resume checkpoint not found: {resume_path} — starting fresh")

    # Baseline F1 for collapse detection
    baseline_f1 = 1.0 / cfg.NUM_CLASSES

    if start_epoch > cfg.NUM_EPOCHS:
        print(f"\n✓ Training already completed ({cfg.NUM_EPOCHS} epochs). "
              f"Use --inference-only to generate predictions.")
        if os.path.exists(best_ckpt_path):
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device,
                                             weights_only=True))
        return history

    print(f"\n{'='*70}")
    print(f"  Training Task {cfg.ACTIVE_TASK} | {cfg.NUM_CLASSES} classes | "
          f"{model.count_parameters():,} params")
    print(f"  Epochs: {start_epoch}→{cfg.NUM_EPOCHS} | Patience: {cfg.PATIENCE} | "
          f"Baseline F1: {baseline_f1:.4f}")
    print(f"  Batch: {cfg.BATCH_SIZE} × {accum_steps} accum = "
          f"{cfg.BATCH_SIZE * accum_steps} effective")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        # Train
        train_loss, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, cfg.USE_AMP, accum_steps=accum_steps)

        # Validate
        val = eval_epoch(model, val_loader, criterion, device, cfg.USE_AMP)

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # History
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val["loss"])
        history["val_f1"].append(val["f1"])
        history["val_bal_acc"].append(val["bal_acc"])
        history["lr"].append(lr)

        # WandB logging (optional)
        try:
            import wandb
            if getattr(wandb, 'run', None) is not None:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/f1": train_f1,
                    "val/loss": val["loss"],
                    "val/f1": val["f1"],
                    "val/bal_acc": val["bal_acc"],
                    "lr": lr,
                })
        except Exception:
            pass

        # Best model checkpoint (just state_dict — always usable for inference)
        saved = ""
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save(model.state_dict(), best_ckpt_path)
            patience_ctr = 0
            saved = " ★ saved"
        else:
            patience_ctr += 1

        # Resume checkpoint (full state — saved every epoch for continuability)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_f1": best_f1,
            "patience_ctr": patience_ctr,
            "history": history,
            "task": cfg.ACTIVE_TASK,
            "num_classes": cfg.NUM_CLASSES,
        }, resume_ckpt_path)

        print(f"Ep {epoch:02d} | "
              f"TrLoss={train_loss:.4f} TrF1={train_f1:.4f} | "
              f"VaLoss={val['loss']:.4f} VaF1={val['f1']:.4f} | "
              f"Baseline={baseline_f1:.4f} | "
              f"LR={lr:.2e} | {elapsed:.1f}s{saved}")

        # Free GPU memory between epochs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        if patience_ctr >= cfg.PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch} (patience={cfg.PATIENCE})")
            break

    # Load best checkpoint
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device,
                                         weights_only=True))
        print(f"\n✓ Loaded best checkpoint (val F1 = {best_f1:.4f})")

    # Final classification report
    val_final = eval_epoch(model, val_loader, criterion, device, cfg.USE_AMP)
    report = classification_report(
        val_final['labels'], val_final['preds'],
        target_names=cfg.LABEL_NAMES, zero_division=0)
    print(f"\n{report}")

    return history
