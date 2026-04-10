"""
Step 2: Extract StarCoder2-3b backbone embeddings (3072-dim) for all data splits.

Saves to cache/:
  {split}_embeddings.pt  - FloatTensor [N, 3072]

The backbone is loaded once per split to keep GPU memory manageable.
Rows are processed in the same order as extract_features.py (same sampling seed).

Usage:
  python extract_embeddings.py                        # all splits
  python extract_embeddings.py --splits train val     # specific splits
  python extract_embeddings.py --batch_size 16        # reduce if OOM
"""

import sys
import yaml
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


def setup_device():
    if not torch.cuda.is_available():
        return torch.device('cpu'), None
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        return torch.device('cuda:1'), {"": 1}
    return torch.device('cuda'), "auto"


def load_backbone(device_map):
    print("Loading StarCoder2-3b (8-bit quantized)...")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    backbone = AutoModel.from_pretrained(
        "bigcode/starcoder2-3b",
        quantization_config=quant_cfg,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    backbone.eval()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Backbone loaded.")
    return backbone, tokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over sequence dimension."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float()
    return (last_hidden_state.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_codes(codes, backbone, tokenizer, device, max_length: int, batch_size: int,
                checkpoint_path: Path = None, checkpoint_every: int = 10000):
    """
    Run all codes through frozen backbone, return [N, hidden_size] float32 tensor.
    Saves a checkpoint every `checkpoint_every` samples so a crash can be resumed.
    """
    start_idx = 0
    all_embeddings = []

    # Resume from checkpoint if available
    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path)
        all_embeddings = list(ckpt['embeddings'])
        start_idx = ckpt['next_idx']
        print(f"  Resuming from checkpoint: {start_idx:,} / {len(codes):,} samples done")

    with torch.no_grad():
        for i in tqdm(range(start_idx, len(codes), batch_size), desc="  Batches",
                      initial=start_idx // batch_size, total=len(codes) // batch_size):
            batch = codes[i : i + batch_size]

            enc = tokenizer(
                batch,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

            input_ids      = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)

            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )

            pooled = mean_pool(outputs.last_hidden_state, attention_mask)
            all_embeddings.extend([e.cpu() for e in pooled])

            # Periodic GPU cache flush
            if (i // batch_size) % 50 == 0:
                torch.cuda.empty_cache()

            # Save checkpoint
            if checkpoint_path and len(all_embeddings) % checkpoint_every < batch_size:
                torch.save({'embeddings': torch.stack(all_embeddings),
                            'next_idx':  i + batch_size}, checkpoint_path)

    # Clean up checkpoint once fully done
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    return torch.stack(all_embeddings)  # [N, 3072]


def extract_split(data_file: str, split: str, cache_dir: Path,
                  max_length: int, batch_size: int):
    print(f"\n{'='*60}")
    print(f"Split: {split}  |  Source: {data_file}")
    print(f"{'='*60}")

    emb_path  = cache_dir / f"{split}_embeddings.pt"
    meta_path = cache_dir / f"{split}_meta.pt"

    if emb_path.exists():
        existing = torch.load(emb_path)
        print(f"Cache exists: {emb_path.name}  (shape {existing.shape}) — skipping.")
        print("  Delete the .pt file to re-extract.")
        return

    # Load the row indices selected by extract_features.py (respects language balancing)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta file not found: {meta_path}\n"
            f"Run extract_features.py --splits {split} first."
        )
    meta = torch.load(meta_path)
    parquet_indices = meta['parquet_indices']
    print(f"Using {len(parquet_indices):,} rows from meta (matches feature extraction order)")

    df = pd.read_parquet(data_file)
    df = df.iloc[parquet_indices].reset_index(drop=True)
    print(f"Selected {len(df):,} rows from parquet")

    codes = df['code'].tolist()

    device, device_map = setup_device()
    print(f"Device: {device}")
    backbone, tokenizer = load_backbone(device_map)

    ckpt_path = cache_dir / f"{split}_embeddings.ckpt.pt"
    print(f"Extracting embeddings (batch_size={batch_size}, max_length={max_length})...")
    print(f"  Checkpoint path: {ckpt_path}  (saved every 10,000 samples)")
    embeddings = embed_codes(codes, backbone, tokenizer, device, max_length, batch_size,
                             checkpoint_path=ckpt_path)
    print(f"Embeddings tensor: {embeddings.shape}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, emb_path)
    print(f"Saved: {emb_path}  ({emb_path.stat().st_size / 1e6:.1f} MB)")

    # Release GPU memory before next split
    del backbone
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract StarCoder2 embeddings for all splits.")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument(
        '--splits', nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test', 'test_sample'],
    )
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Reduce to 32 or 16 if you get OOM errors')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cache_dir  = Path(config['data']['cache_dir'])
    max_length = config['data']['max_length']

    split_files = {
        'train':       config['data']['train_file'],
        'val':         config['data']['val_file'],
        'test':        config['data']['test_file'],
        'test_sample': config['data']['test_sample_file'],
    }

    for split in args.splits:
        data_file = split_files[split]
        if Path(data_file).exists():
            extract_split(data_file, split, cache_dir, max_length, args.batch_size)
        else:
            print(f"\nSkipping {split}: file not found at {data_file}")

    print("\n" + "="*60)
    print("Embedding extraction complete!")
    print(f"Cache directory: {cache_dir.resolve()}")
    print("\nNext step: python train.py")


if __name__ == '__main__':
    main()
