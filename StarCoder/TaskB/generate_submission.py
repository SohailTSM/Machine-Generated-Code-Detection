"""
Step 5: Generate predictions for the test set and write submission.csv.

The output format matches sample_submission.csv:
  ID,label
  12345,3
  ...

Labels:
  0 = human
  1 = deepseek-ai
  2 = qwen
  3 = 01-ai
  4 = bigcode
  5 = gemma
  6 = phi
  7 = meta-llama
  8 = ibm-granite
  9 = mistral
  10 = openai

Usage:
  python generate_submission.py
  python generate_submission.py --checkpoint checkpoints/epoch_3.pt
  python generate_submission.py --output my_submission.csv
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.cuda.amp as amp
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TaskBModel
from dataset import CachedDataset


def get_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        return torch.device('cuda:1')
    return torch.device('cuda')


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_ids, all_preds = [], []

    for batch in tqdm(loader, desc="Predicting"):
        feats  = batch['handcrafted_features'].to(device, non_blocking=True)
        embeds = batch['backbone_embedding'].to(device, non_blocking=True)

        with amp.autocast(dtype=torch.bfloat16):
            out = model(handcrafted_features=feats, backbone_embedding=embeds)

        preds = out['logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_ids.extend(batch['id'] if isinstance(batch['id'], list) else batch['id'].tolist())

    return all_ids, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--output',     default=None,
                        help='Output CSV path (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (default: 2x training batch)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    class_names_map = config['model'].get('class_names', {})
    num_labels      = config['model']['num_labels']
    class_names     = [class_names_map.get(i, str(i)) for i in range(num_labels)]

    device = get_device()
    print(f"Device: {device}")

    output_path = args.output or config['output']['submission_file']

    # Dataset
    print("\nLoading test split from cache...")
    test_ds = CachedDataset(config['data']['cache_dir'], split='test')

    batch_size = args.batch_size or config['training']['batch_size'] * 2
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,    # Must NOT shuffle — keep ID order
        num_workers=0,
        pin_memory=True,
    )

    # Model
    print(f"\nLoading model from: {args.checkpoint}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run: python train.py  first."
        )

    model = TaskBModel(config)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', {}))
    model.load_state_dict(state, strict=False)
    model.eval()

    if 'val_f1' in ckpt:
        print(f"  Checkpoint val macro F1: {ckpt['val_f1']:.4f}")
    if 'epoch' in ckpt:
        print(f"  Checkpoint epoch: {ckpt['epoch'] + 1}")

    # Predict
    print(f"\nRunning inference on {len(test_ds):,} test samples...")
    ids, preds = predict(model, loader, device)

    # Build and save submission CSV
    submission = pd.DataFrame({'ID': ids, 'label': preds})
    submission = submission.sort_values('ID').reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    # Summary
    label_counts = submission['label'].value_counts().sort_index().to_dict()
    print(f"\nSubmission saved: {output_path}")
    print(f"  Total predictions: {len(submission):,}")
    print("  Label distribution:")
    for i in range(num_labels):
        count = label_counts.get(i, 0)
        pct   = count / len(submission) * 100 if len(submission) > 0 else 0
        print(f"    {i:2d} ({class_names[i]:<15}): {count:>8,}  ({pct:.1f}%)")

    print(f"\nFirst 5 rows:")
    print(submission.head().to_string(index=False))


if __name__ == '__main__':
    main()
