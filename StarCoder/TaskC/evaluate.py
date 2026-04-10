"""
Evaluate a trained Task C model on the validation set.

Usage:
  python evaluate.py                                    # val set, best_model.pt
  python evaluate.py --checkpoint checkpoints/epoch_3.pt
  python evaluate.py --split val
"""

import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from model import TaskCModel
from dataset import CachedDataset


def get_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        return torch.device('cuda:1')
    return torch.device('cuda')


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        feats  = batch['handcrafted_features'].to(device, non_blocking=True)
        embeds = batch['backbone_embedding'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with amp.autocast(dtype=torch.bfloat16):
            out = model(handcrafted_features=feats, backbone_embedding=embeds, labels=labels)

        probs = torch.softmax(out['logits'].float(), dim=-1)
        preds = out['logits'].argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--split',      default='val',
                        choices=['val'],
                        help='Which split to evaluate (must have labels)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build ordered class names list from config
    class_names_map = config['model'].get('class_names', {})
    num_labels      = config['model']['num_labels']
    class_names     = [class_names_map.get(i, str(i)) for i in range(num_labels)]

    device = get_device()
    print(f"Device: {device}")

    # Dataset
    print(f"\nLoading {args.split} split from cache...")
    ds = CachedDataset(config['data']['cache_dir'], split=args.split)
    if not ds.has_labels:
        raise ValueError(f"Split '{args.split}' has no labels — cannot evaluate.")

    loader = DataLoader(
        ds,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model + checkpoint
    print(f"\nLoading model from: {args.checkpoint}")
    model = TaskCModel(config)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', {}))
    model.load_state_dict(state, strict=False)
    model.eval()

    if 'val_f1' in ckpt:
        print(f"  Checkpoint val macro F1 (train time): {ckpt['val_f1']:.4f}")
    if 'epoch' in ckpt:
        print(f"  Checkpoint epoch: {ckpt['epoch'] + 1}")

    # Run
    preds, labels, probs = run_inference(model, loader, device)

    # Metrics
    acc         = accuracy_score(labels, preds)
    f1_macro    = f1_score(labels, preds, average='macro',    zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS  ({args.split})")
    print("="*60)
    print(f"Accuracy:          {acc:.4f}  ({acc*100:.2f}%)")
    print(f"F1 Macro:          {f1_macro:.4f}   ← official metric")
    print(f"F1 Weighted:       {f1_weighted:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    header = "         " + "".join(f"  Pred {i} ({class_names[i][:8]:<8})" for i in range(num_labels))
    print(header)
    for i in range(num_labels):
        row = f"True {i} ({class_names[i][:8]:<8})" + "".join(f"  {cm[i][j]:>12,}" for j in range(num_labels))
        print(row)

    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # Confidence
    conf = probs.max(axis=1)
    print(f"Prediction confidence — mean: {conf.mean():.4f}  "
          f"median: {np.median(conf):.4f}  "
          f"min: {conf.min():.4f}")

    # Save predictions
    out_dir = Path('evaluation_results')
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f'predictions_{args.split}.npz',
             predictions=preds, labels=labels, probabilities=probs)
    print(f"\nPredictions saved to: {out_dir / f'predictions_{args.split}.npz'}")


if __name__ == '__main__':
    main()
