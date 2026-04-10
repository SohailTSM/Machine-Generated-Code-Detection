"""
Step 1: Extract handcrafted features (102-dim) for all data splits.

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
        self.pattern = EnhancedPatternDetector()
        self.ast = EnhancedASTExtractor()

    def extract(self, code: str) -> torch.Tensor:
        try:
            style = self.stylometric.extract(code)   # 12
            pattern = self.pattern.extract(code)      # 57
            ast = self.ast.extract(code)              # 33

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


def balance_languages(df: pd.DataFrame, max_samples: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Cap each language at max_samples_per_language so no single language dominates,
    while keeping all data from minority languages intact.
    Falls back to min-count equalisation only when max_samples is not set.
    Returns a shuffled DataFrame with the original index preserved.
    """
    counts  = df['language'].value_counts()
    n_langs = len(counts)

    if max_samples:
        # Each language gets up to max_samples / n_langs rows (cap majority, keep minority)
        cap = max_samples // n_langs
    else:
        # Fall back to equal counts across languages (original behaviour)
        cap = int(counts.min())

    print(f"Language capping  ({n_langs} languages, cap={cap:,}/lang):")
    total = 0
    groups = []
    for lang, grp in df.groupby('language'):
        n = min(len(grp), cap)
        groups.append(grp.sample(n=n, random_state=seed))
        print(f"  {lang:<20} {len(grp):>8,} → {n:,}")
        total += n
    print(f"  {'TOTAL':<20} {len(df):>8,} → {total:,}")

    balanced = pd.concat(groups).sample(frac=1, random_state=seed)
    return balanced


def extract_split(data_file: str, split: str, cache_dir: Path,
                  balance: bool = False, max_samples: int = None, n_workers: int = 10):
    """Extract and cache features for one split. Skips if cache already exists."""
    print(f"\n{'='*60}")
    print(f"Split: {split}  |  Source: {data_file}")
    print(f"{'='*60}")

    feat_path = cache_dir / f"{split}_features.pt"
    meta_path = cache_dir / f"{split}_meta.pt"

    if feat_path.exists() and meta_path.exists():
        existing = torch.load(feat_path)
        print(f"Cache exists: {feat_path.name}  (shape {existing.shape}) — skipping.")
        print("  Delete the .pt files to re-extract.")
        return

    # Load parquet
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df):,} samples")

    # Language balancing (only for labelled splits; test has no language label to balance on)
    if balance and 'language' in df.columns:
        df = balance_languages(df, max_samples=max_samples)
    elif max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {max_samples:,} rows (random_state=42)")

    df = df.reset_index(drop=True)
    codes = df['code'].tolist()
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

    # Build metadata dict — also stores row order so extract_embeddings.py
    # can reproduce the exact same sample without re-reading the parquet.
    meta = {
        'num_samples': len(df),
        'source_file': str(data_file),
        'balanced':    balance,
        # Preserve original parquet indices so embeddings can be extracted
        # for the identical rows in the identical order.
        'parquet_indices': df.index.tolist(),
    }

    # IDs: test.parquet has an explicit 'ID' column; train/val use the index
    if 'ID' in df.columns:
        meta['ids'] = df['ID'].tolist()
    else:
        meta['ids'] = df.index.tolist()

    # Labels present in train / val / test_sample, absent in test
    if 'label' in df.columns:
        meta['labels'] = torch.tensor(df['label'].tolist(), dtype=torch.long)
        counts = df['label'].value_counts().sort_index().to_dict()
        print(f"Labels: {counts}  (0=Human, 1=AI)")

    if 'language' in df.columns:
        lang_counts = df['language'].value_counts().sort_index().to_dict()
        print(f"Languages: { {k: v for k, v in lang_counts.items()} }")

    # Save
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, feat_path)
    torch.save(meta, meta_path)
    print(f"Saved: {feat_path}  ({feat_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract handcrafted features for all splits.")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument(
        '--splits', nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test', 'test_sample'],
        help='Which splits to process'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of CPU cores (overrides config; default: feature_extraction_workers)'
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_dir = Path(config['data']['cache_dir'])
    n_workers = args.workers or config['data'].get('feature_extraction_workers', 10)
    balance   = config['data'].get('balance_by_language', False)

    # test split is never balanced — it has no labels and we must predict on all rows
    split_cfg = {
        'train':       (config['data']['train_file'],       balance, config['data'].get('max_train_samples')),
        'val':         (config['data']['val_file'],         balance, config['data'].get('max_val_samples')),
        'test':        (config['data']['test_file'],        False,   None),
        'test_sample': (config['data']['test_sample_file'], False,   None),
    }

    for split in args.splits:
        data_file, do_balance, max_samples = split_cfg[split]
        if Path(data_file).exists():
            extract_split(data_file, split, cache_dir, do_balance, max_samples, n_workers)
        else:
            print(f"\nSkipping {split}: file not found at {data_file}")

    print("\n" + "="*60)
    print("Feature extraction complete!")
    print(f"Cache directory: {cache_dir.resolve()}")
    print("\nNext step: python extract_embeddings.py")


if __name__ == '__main__':
    main()
