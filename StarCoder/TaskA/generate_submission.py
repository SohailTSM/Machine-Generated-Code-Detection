"""
Step 5: Generate predictions for the test set and write submission.csv.

The output format matches sample_submission.csv:
  ID,label
  12345,0
  ...

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

from model import TaskAModel
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

    device = get_device()
    print(f"Device: {device}")

    output_path = args.output or config['output']['submission_file']

    # Dataset
    print(f"\nLoading test split from cache...")
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

    model = TaskAModel(config)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', {}))
    model.load_state_dict(state, strict=False)
    model.eval()

    if 'val_acc' in ckpt:
        print(f"  Checkpoint val accuracy: {ckpt['val_acc']:.4f}")
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
    print(f"  Label distribution: 0 (Human)={label_counts.get(0, 0):,}  "
          f"1 (AI)={label_counts.get(1, 0):,}")
    print(f"\nFirst 5 rows:")
    print(submission.head().to_string(index=False))


if __name__ == '__main__':
    main()
