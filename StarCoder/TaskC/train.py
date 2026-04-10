"""
Step 3: Train the hybrid StarCoder2 + handcrafted-features classifier for Task C.

Only the classification head is trained (~8M params).
The backbone weights stay frozen throughout.

Key differences from Task A:
  - 4-class output (human, machine, hybrid, adversarial)
  - Class-weighted CrossEntropyLoss (handles imbalance: 485K human vs 85K-210K others)
  - Best model tracked by val macro-F1 (official competition metric)

Requires: extract_features.py and extract_embeddings.py to have been run.

Usage:
  python train.py
  python train.py --no-auto-resume    # start from scratch, ignore checkpoints
  python train.py --resume checkpoints/epoch_2.pt
"""

import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

from model import TaskCModel
from dataset import CachedDataset


# ── helpers ──────────────────────────────────────────────────────────────────

def get_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        return torch.device('cuda:1')
    return torch.device('cuda')


def compute_class_weights(cache_dir: str, num_labels: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training label distribution.
    Returns a float32 tensor of shape [num_labels].
    """
    meta_path = Path(cache_dir) / "train_meta.pt"
    if not meta_path.exists():
        print("  Warning: train_meta.pt not found — using uniform class weights.")
        return torch.ones(num_labels, dtype=torch.float32)

    meta = torch.load(meta_path, weights_only=False)
    if 'labels' not in meta:
        print("  Warning: no labels in meta — using uniform class weights.")
        return torch.ones(num_labels, dtype=torch.float32)

    labels = meta['labels']
    counts = torch.zeros(num_labels, dtype=torch.float32)
    for c in range(num_labels):
        counts[c] = (labels == c).sum().item()

    # Replace zeros with 1 to avoid division-by-zero (unseen classes)
    counts = counts.clamp(min=1)

    # Inverse frequency, normalised so the mean weight = 1
    weights = 1.0 / counts
    weights = weights / weights.mean()

    class_names = {0: 'human', 1: 'machine', 2: 'hybrid', 3: 'adversarial'}
    print("  Class weights (inverse-frequency, mean-normalised):")
    for c, (cnt, w) in enumerate(zip(counts.tolist(), weights.tolist())):
        name = class_names.get(c, str(c))
        print(f"    class {c:2d} ({name:<11}): {int(cnt):>8,} samples  →  weight {w:.4f}")

    return weights


def train_epoch(model, loader, optimizer, scheduler, scaler, config, device, class_weights):
    model.train()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    accum = config['training']['gradient_accumulation_steps']
    optimizer.zero_grad()

    bar = tqdm(loader, desc="Train", mininterval=2.0)
    for step, batch in enumerate(bar):
        feats  = batch['handcrafted_features'].to(device, non_blocking=True)
        embeds = batch['backbone_embedding'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with amp.autocast(dtype=torch.bfloat16):
            out  = model(handcrafted_features=feats, backbone_embedding=embeds,
                         labels=labels, class_weights=class_weights)
            loss = out['loss'] / accum

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  NaN/Inf loss at step {step} — skipping batch")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum
        n_batches  += 1

        preds = out['logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if n_batches % 20 == 0:
            bar.set_postfix({'loss': f"{total_loss / n_batches:.4f}"})

    avg_loss  = total_loss / max(n_batches, 1)
    macro_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1


@torch.no_grad()
def validate(model, loader, device, class_weights):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Val", mininterval=2.0):
        feats  = batch['handcrafted_features'].to(device, non_blocking=True)
        embeds = batch['backbone_embedding'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with amp.autocast(dtype=torch.bfloat16):
            out = model(handcrafted_features=feats, backbone_embedding=embeds,
                        labels=labels, class_weights=class_weights)

        if not torch.isnan(out['loss']):
            total_loss += out['loss'].item()
            n_batches  += 1

        preds = out['logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, macro_f1, acc


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_f1, train_loss, val_loss):
    """Save only trainable head parameters (not the frozen backbone)."""
    state = {k: v for k, v in model.state_dict().items() if 'backbone' not in k}
    torch.save({
        'epoch':      epoch,
        'state_dict': state,
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'val_f1':     val_f1,
        'train_loss': train_loss,
        'val_loss':   val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', {}))
    model.load_state_dict(state, strict=False)
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if 'scheduler' in ckpt and ckpt['scheduler']:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0) + 1, ckpt.get('val_f1', 0.0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',         default='config.yaml')
    parser.add_argument('--resume',         default=None,  help='Path to checkpoint to resume from')
    parser.add_argument('--no-auto-resume', action='store_true', help='Ignore existing checkpoints')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")
        torch.backends.cudnn.benchmark         = True
        torch.backends.cuda.matmul.allow_tf32  = True
        torch.backends.cudnn.allow_tf32        = True

    # ── class weights ─────────────────────────────────────────────────────
    num_labels = config['model']['num_labels']
    if config['training'].get('use_class_weights', True):
        print("\nComputing class weights...")
        class_weights = compute_class_weights(config['data']['cache_dir'], num_labels)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
        print("\nClass weights disabled.")

    # ── datasets & loaders ───────────────────────────────────────────────
    print("\nLoading datasets from cache...")
    train_ds = CachedDataset(config['data']['cache_dir'], split='train')
    val_ds   = CachedDataset(config['data']['cache_dir'], split='val')

    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── model ────────────────────────────────────────────────────────────
    print("\nCreating model...")
    model = TaskCModel(config)

    for p in model.backbone.parameters():
        p.requires_grad = False

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:  total={total_p:,}  trainable={trainable_p:,}  "
          f"frozen={total_p - trainable_p:,}")

    # ── optimizer & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    total_steps  = (len(train_loader) * config['training']['num_epochs']
                    // config['training']['gradient_accumulation_steps'])
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = amp.GradScaler()

    # ── resume ───────────────────────────────────────────────────────────
    ckpt_dir    = Path(config['output']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val_f1 = 0.0

    resume_path = args.resume
    if resume_path is None and not args.no_auto_resume:
        epoch_ckpts = sorted(ckpt_dir.glob('epoch_*.pt'),
                             key=lambda p: int(p.stem.split('_')[1]),
                             reverse=True)
        if epoch_ckpts:
            resume_path = str(epoch_ckpts[0])
            print(f"\nAuto-detected checkpoint: {resume_path}")
            print("  (use --no-auto-resume to start from scratch)")

    if resume_path:
        print(f"Loading checkpoint: {resume_path}")
        start_epoch, best_val_f1 = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        print(f"Resuming from epoch {start_epoch}, best val F1: {best_val_f1:.4f}")

    # ── training loop ─────────────────────────────────────────────────────
    num_epochs = config['training']['num_epochs']
    print(f"\nTraining for {num_epochs} epochs (start epoch: {start_epoch + 1})...")
    print(f"Target metric: Macro F1  (official SemEval-2026 Task 13 metric)\n")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} / {num_epochs}")
        print(f"{'='*60}")

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, device, class_weights
        )
        print(f"Train  Loss: {train_loss:.4f}   Macro F1: {train_f1:.4f}")

        val_loss, val_f1, val_acc = validate(model, val_loader, device, class_weights)
        print(f"Val    Loss: {val_loss:.4f}   Macro F1: {val_f1:.4f}   Accuracy: {val_acc:.4f}")

        # Best model tracked by macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path   = ckpt_dir / "best_model.pt"
            save_checkpoint(best_path, model, optimizer, scheduler,
                            epoch, val_f1, train_loss, val_loss)
            print(f"  New best saved: {best_path}  (macro F1 = {best_val_f1:.4f})")

        # Epoch checkpoint
        epoch_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
        save_checkpoint(epoch_path, model, optimizer, scheduler,
                        epoch, val_f1, train_loss, val_loss)
        print(f"  Epoch checkpoint: {epoch_path}")

    print(f"\n{'='*60}")
    print(f"Training complete.  Best val Macro F1: {best_val_f1:.4f}")
    print(f"Best model: {ckpt_dir / 'best_model.pt'}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  python evaluate.py                   # evaluate on validation set")
    print("  python generate_submission.py        # generate test submission CSV")


if __name__ == '__main__':
    main()
