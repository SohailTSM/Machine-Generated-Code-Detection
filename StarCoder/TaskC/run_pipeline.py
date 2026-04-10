"""
Full pipeline runner for Task C.

Runs all steps in order:
  1. extract_features.py   — handcrafted features for train / val / test
  2. extract_embeddings.py — StarCoder2 embeddings for train / val / test
  3. train.py              — train the hybrid classifier (4-class hybrid detection)
  4. evaluate.py           — evaluate on validation set
  5. generate_submission.py — write submission.csv for test set

Usage:
  python run_pipeline.py                        # all steps
  python run_pipeline.py --start-from 3        # resume from training
  python run_pipeline.py --steps 1 2           # only extraction steps
  python run_pipeline.py --no-auto-resume      # train from scratch (step 3)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Always run every subprocess from TaskC/ regardless of where the user invoked this script
PIPELINE_DIR = Path(__file__).resolve().parent


STEPS = [
    (1, "Feature extraction",    ["python", "extract_features.py"]),
    (2, "Embedding extraction",  ["python", "extract_embeddings.py"]),
    (3, "Training",              ["python", "train.py"]),
    (4, "Evaluation",            ["python", "evaluate.py"]),
    (5, "Submission generation", ["python", "generate_submission.py"]),
]


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_step(name: str, cmd: list[str]) -> bool:
    """Run a subprocess step. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, check=False, cwd=PIPELINE_DIR)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  Completed in {fmt_duration(elapsed)}")
        return True
    else:
        print(f"\n  FAILED (exit code {result.returncode}) after {fmt_duration(elapsed)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full Task C pipeline.")
    parser.add_argument(
        '--steps', nargs='+', type=int,
        metavar='N',
        help='Run only these step numbers (e.g. --steps 1 2)'
    )
    parser.add_argument(
        '--start-from', type=int, default=1, metavar='N',
        help='Start from step N (skip earlier steps)'
    )
    parser.add_argument(
        '--no-auto-resume', action='store_true',
        help='Pass --no-auto-resume to train.py (start training from scratch)'
    )
    args = parser.parse_args()

    # Resolve which steps to run
    if args.steps:
        selected = set(args.steps)
    else:
        selected = {n for n, _, _ in STEPS if n >= args.start_from}

    # Patch train command if --no-auto-resume requested
    steps_to_run = []
    for n, name, cmd in STEPS:
        if n not in selected:
            continue
        if n == 3 and args.no_auto_resume:
            cmd = cmd + ['--no-auto-resume']
        steps_to_run.append((n, name, cmd))

    if not steps_to_run:
        print("No steps selected.")
        sys.exit(0)

    print(f"\nPipeline: Task C (4-class hybrid detection) — {len(steps_to_run)} step(s) selected")
    print(f"Working directory: {PIPELINE_DIR}")
    for n, name, _ in steps_to_run:
        print(f"  Step {n}: {name}")

    pipeline_start = time.time()
    failed_step    = None

    for n, name, cmd in steps_to_run:
        ok = run_step(f"Step {n}/{len(STEPS)}: {name}", cmd)
        if not ok:
            failed_step = (n, name)
            break

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")

    if failed_step:
        print(f"  Pipeline STOPPED at Step {failed_step[0]}: {failed_step[1]}")
        print(f"  Fix the error above, then re-run with:  --start-from {failed_step[0]}")
        print(f"  Total time before failure: {fmt_duration(total)}")
        print(f"{'='*60}")
        sys.exit(1)
    else:
        print(f"  Pipeline COMPLETE  ({fmt_duration(total)})")
        print(f"{'='*60}")
        print("\nOutputs:")
        print("  checkpoints/best_model.pt   — trained model")
        print("  evaluation_results/          — val metrics")
        print("  submission.csv               — test predictions")


if __name__ == '__main__':
    main()
