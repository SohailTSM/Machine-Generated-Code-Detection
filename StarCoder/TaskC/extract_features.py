"""
Step 1: Extract handcrafted features (102-dim) for all data splits.

Uses the same UnifiedFeatureExtractor from MyProject/src/features/:
  12 stylometric + 57 pattern + 33 AST = 102 features

Saves to cache/:
  {split}_features.pt  - FloatTensor [N, 102]
  {split}_meta.pt      - dict with 'ids', 'labels' (if available), 'num_samples'

Usage:
  python extract_features.py                        # all splits
  python extract_features.py --splits train val     # specific splits
  python extract_features.py --splits test          # test only
"""

import sys
import yaml
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append(str(Path(__file__).parent.parent))
from features.stylometric import StylometricExtractor
from features.pattern_detector import EnhancedPatternDetector
from features.ast_extractor import EnhancedASTExtractor


class UnifiedFeatureExtractor:
    """Extracts 102-dim handcrafted features: 12 stylometric + 57 pattern + 33 AST."""

    def __init__(self):
        self.stylometric = StylometricExtractor()
        self.pattern     = EnhancedPatternDetector()
        self.ast         = EnhancedASTExtractor()

    def extract(self, code: str) -> torch.Tensor:
        try:
            style   = self.stylometric.extract(code)   # 12
            pattern = self.pattern.extract(code)        # 57
            ast     = self.ast.extract(code)            # 33

            feats = torch.cat([
                torch.tensor(style,   dtype=torch.float32),
                torch.tensor(pattern, dtype=torch.float32),
                torch.tensor(ast,     dtype=torch.float32),
            ])

            if feats.shape[0] != 102:
                return torch.zeros(102, dtype=torch.float32)

            return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return torch.zeros(102, dtype=torch.float32)


def _worker(code: str) -> torch.Tensor:
    """Per-process worker — creates a fresh extractor to avoid shared-state issues."""
    extractor = UnifiedFeatureExtractor()
    return extractor.extract(code)


def extract_split(data_file: str, split: str, cache_dir: Path,
                  max_samples: int = None, n_workers: int = 10):
    """Extract and cache features for one split. Skips if cache already exists."""
    print(f"\n{'='*60}")
    print(f"Split: {split}  |  Source: {data_file}")
    print(f"{'='*60}")

    feat_path = cache_dir / f"{split}_features.pt"
    meta_path = cache_dir / f"{split}_meta.pt"

    if feat_path.exists() and meta_path.exists():
        existing = torch.load(feat_path, weights_only=True)
        print(f"Cache exists: {feat_path.name}  (shape {existing.shape}) — skipping.")
        print("  Delete the .pt files to re-extract.")
        return

    # Load parquet
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df):,} samples")

    # Optional cap
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {max_samples:,} rows (random_state=42)")

    df = df.reset_index(drop=True)
    codes   = df['code'].tolist()
    n_cores = min(n_workers, cpu_count())
    print(f"Extracting features with {n_cores} CPU cores (of {cpu_count()} available)...")

    with Pool(processes=n_cores, maxtasksperchild=500) as pool:
        features = list(tqdm(
            pool.imap(_worker, codes, chunksize=100),
            total=len(codes),
            desc=f"  {split}",
        ))

    features = torch.stack(features)  # [N, 102]
    print(f"Features tensor: {features.shape}")

    # Build metadata dict
    meta = {
        'num_samples':     len(df),
        'source_file':     str(data_file),
        'parquet_indices': df.index.tolist(),
    }

    # IDs: test.parquet has an explicit 'ID' column; train/val use the index
    if 'ID' in df.columns:
        meta['ids'] = df['ID'].tolist()
    else:
        meta['ids'] = df.index.tolist()

    # Labels present in train / val, absent in test
    if 'label' in df.columns:
        meta['labels'] = torch.tensor(df['label'].tolist(), dtype=torch.long)
        unique, counts = meta['labels'].unique(return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"Labels: {dist}  (0=human 1=machine 2=hybrid 3=adversarial)")

    if 'language' in df.columns:
        lang_counts = df['language'].value_counts().sort_index().to_dict()
        print(f"Languages: {lang_counts}")

    # Save
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, feat_path)
    torch.save(meta,     meta_path)
    print(f"Saved: {feat_path}  ({feat_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract handcrafted features for Task C splits.")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument(
        '--splits', nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='Which splits to process'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of CPU cores (overrides config)'
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_dir = Path(config['data']['cache_dir'])
    n_workers = args.workers or config['data'].get('feature_extraction_workers', 10)

    split_cfg = {
        'train': (config['data']['train_file'], config['data'].get('max_train_samples')),
        'val':   (config['data']['val_file'],   config['data'].get('max_val_samples')),
        'test':  (config['data']['test_file'],  None),
    }

    for split in args.splits:
        data_file, max_samples = split_cfg[split]
        if Path(data_file).exists():
            extract_split(data_file, split, cache_dir, max_samples, n_workers)
        else:
            print(f"\nSkipping {split}: file not found at {data_file}")
            print(f"  Update '{split}_file' in config.yaml once you download the data.")

    print("\n" + "="*60)
    print("Feature extraction complete!")
    print(f"Cache directory: {cache_dir.resolve()}")
    print("\nNext step: python extract_embeddings.py")


if __name__ == '__main__':
    main()
