"""
Language-agnostic feature extraction for code snippets.

All features are computed with basic string operations and regex — no
language-specific parsers required.  This ensures the features generalise
to C++, Python, Java, Go, PHP, C#, C, and JavaScript without modification.

Features are grouped into categories:
  1. Length statistics
  2. Whitespace / indentation
  3. Lexical / token-level
  4. Comment patterns
  5. Structural proxies
  6. Punctuation ratios
  7. Entropy
  8. Repetition
"""

import os
import re
import math
from collections import Counter
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd

from src.config import NUM_WORKERS, FEATURES_DIR, SEED
from src.utils import get_logger

logger = get_logger(__name__, log_file="features.log")

# Bump whenever feature definitions change to avoid stale cached matrices.
FEATURE_SET_VERSION = "v2"


# ======================================================================
# Individual feature functions
# ======================================================================

def _safe_std(values) -> float:
    """Standard deviation, returning 0.0 for empty / single-item sequences."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _entropy(counter: Counter) -> float:
    """Shannon entropy (bits) from a Counter of items."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ------------------------------------------------------------------
# 1. Length statistics
# ------------------------------------------------------------------
def _length_features(code: str, lines: list[str]) -> dict:
    line_lens = [len(l) for l in lines]
    return {
        "total_chars": len(code),
        "total_lines": len(lines),
        "avg_line_length": float(np.mean(line_lens)) if line_lens else 0.0,
        "max_line_length": max(line_lens) if line_lens else 0,
        "std_line_length": _safe_std(line_lens),
        "median_line_length": float(np.median(line_lens)) if line_lens else 0.0,
    }


# ------------------------------------------------------------------
# 2. Whitespace / indentation
# ------------------------------------------------------------------
def _whitespace_features(lines: list[str]) -> dict:
    # Leading spaces per line (indentation proxy)
    indents = []
    blank_count = 0
    trailing_ws = 0
    for line in lines:
        stripped = line.lstrip(" \t")
        indent = len(line) - len(stripped)
        indents.append(indent)
        if stripped == "":
            blank_count += 1
        if line != line.rstrip():
            trailing_ws += 1

    n = max(len(lines), 1)
    return {
        "avg_indent": float(np.mean(indents)) if indents else 0.0,
        "max_indent": max(indents) if indents else 0,
        "std_indent": _safe_std(indents),
        "blank_line_ratio": blank_count / n,
        "trailing_ws_ratio": trailing_ws / n,
    }


# ------------------------------------------------------------------
# 3. Lexical / token-level
# ------------------------------------------------------------------
# Simple tokeniser: split on whitespace + punctuation boundaries
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\s\w]")

_COMMON_KEYWORDS = {
    "if", "else", "elif", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "try", "catch", "finally", "throw", "class", "def", "function", "func", "fn",
    "import", "from", "include", "package", "namespace", "public", "private", "static", "new",
}

_CTRL_RE = re.compile(r"\b(if|else|elif|for|while|switch|case|do|try|catch)\b", re.IGNORECASE)
_LOOP_RE = re.compile(r"\b(for|while|do)\b", re.IGNORECASE)
_COND_RE = re.compile(r"\b(if|elif|else|switch|case)\b", re.IGNORECASE)

def _lexical_features(code: str) -> dict:
    tokens = _TOKEN_RE.findall(code)
    n = max(len(tokens), 1)

    # Identifiers = tokens that are purely alpha/underscore (not numbers, not
    # single-char punctuation)
    identifiers = [t for t in tokens if re.match(r"^[A-Za-z_]\w*$", t)]
    id_lens = [len(i) for i in identifiers]

    numbers = [t for t in tokens if re.match(r"^[0-9]", t)]
    keyword_count = sum(1 for t in tokens if t.lower() in _COMMON_KEYWORDS)
    keyword_to_identifier_ratio = keyword_count / max(len(identifiers), 1)

    return {
        "total_tokens": len(tokens),
        "unique_token_ratio": len(set(tokens)) / n,
        "avg_identifier_length": float(np.mean(id_lens)) if id_lens else 0.0,
        "std_identifier_length": _safe_std(id_lens),
        "number_ratio": len(numbers) / n,
        "identifier_count": len(identifiers),
        "keyword_count": keyword_count,
        "keyword_to_identifier_ratio": keyword_to_identifier_ratio,
    }


# ------------------------------------------------------------------
# 4. Comment patterns
# ------------------------------------------------------------------
# Matches single-line comments: // ... or # ...
_SINGLE_COMMENT_RE = re.compile(r"^\s*(//|#)", re.MULTILINE)
# Inline comments at end of a code line
_INLINE_COMMENT_RE = re.compile(r"\S.*(//|#)\s*\S")

def _comment_features(code: str, lines: list[str]) -> dict:
    n = max(len(lines), 1)
    comment_lines = len(_SINGLE_COMMENT_RE.findall(code))
    inline_comments = len(_INLINE_COMMENT_RE.findall(code))

    # Very rough multi-line comment count (/* ... */ or """ ... """)
    multi_line = len(re.findall(r"/\*.*?\*/", code, re.DOTALL))
    multi_line += len(re.findall(r'""".*?"""', code, re.DOTALL))
    multi_line += len(re.findall(r"'''.*?'''", code, re.DOTALL))

    total_comments = comment_lines + multi_line
    return {
        "comment_line_ratio": comment_lines / n,
        "inline_comment_ratio": inline_comments / n,
        "multi_line_comment_count": multi_line,
        "comment_to_code_ratio": total_comments / n,
    }


# ------------------------------------------------------------------
# 5. Structural proxies
# ------------------------------------------------------------------
def _structural_features(code: str, lines: list[str]) -> dict:
    # Nesting depth proxy: count braces/indentation levels
    max_depth = 0
    cur_depth = 0
    depths = []
    for char in code:
        if char == "{":
            cur_depth += 1
            max_depth = max(max_depth, cur_depth)
        elif char == "}":
            cur_depth = max(cur_depth - 1, 0)
        depths.append(cur_depth)

    # Function-like declarations (rough)
    func_count = len(re.findall(
        r"\b(?:def|function|func|void|int|string|bool|public|private|static)\s+\w+\s*\(",
        code,
    ))

    # Import / include statements
    import_count = len(re.findall(
        r"^\s*(?:import|from|#include|using|require|require_once)\b",
        code,
        re.MULTILINE,
    ))

    return {
        "max_brace_depth": max_depth,
        "avg_brace_depth": float(np.mean(depths)) if depths else 0.0,
        "func_count": func_count,
        "import_count": import_count,
    }


def _control_flow_features(code: str) -> dict:
    """Language-agnostic control-flow proxies for generation patterns."""
    ctrl_count = len(_CTRL_RE.findall(code))
    loop_count = len(_LOOP_RE.findall(code))
    cond_count = len(_COND_RE.findall(code))

    tokens = _TOKEN_RE.findall(code)
    token_count = max(len(tokens), 1)

    # Recursion proxy: function name appears in both definition and call contexts.
    func_names = re.findall(r"\b(?:def|function|func|fn)\s+([A-Za-z_]\w*)\s*\(", code, re.IGNORECASE)
    recursive_like = 0
    for name in func_names:
        # One hit is the definition itself; more than one suggests self-call presence.
        if len(re.findall(rf"\b{name}\s*\(", code)) > 1:
            recursive_like += 1

    # Competitive-programming / templated code proxy patterns.
    template_like = 0
    template_patterns = [
        r"#include\s*<bits/stdc\+\+\.h>",
        r"using\s+namespace\s+std",
        r"if\s+__name__\s*==\s*['\"]__main__['\"]",
        r"public\s+static\s+void\s+main",
        r"class\s+Solution",
    ]
    for pat in template_patterns:
        template_like += len(re.findall(pat, code, re.IGNORECASE))

    return {
        "control_flow_count": ctrl_count,
        "loop_count": loop_count,
        "conditional_count": cond_count,
        "control_flow_density": ctrl_count / token_count,
        "recursive_like_count": recursive_like,
        "template_like_count": template_like,
    }


# ------------------------------------------------------------------
# 6. Punctuation ratios
# ------------------------------------------------------------------
def _punctuation_features(code: str) -> dict:
    n = max(len(code), 1)
    return {
        "brace_ratio": (code.count("{") + code.count("}")) / n,
        "bracket_ratio": (code.count("[") + code.count("]")) / n,
        "paren_ratio": (code.count("(") + code.count(")")) / n,
        "semicolon_density": code.count(";") / n,
        "comma_density": code.count(",") / n,
        "operator_diversity": len(set(re.findall(r"[+\-*/=<>!&|^%]+", code))),
    }


# ------------------------------------------------------------------
# 7. Entropy
# ------------------------------------------------------------------
def _entropy_features(code: str) -> dict:
    char_counts = Counter(code)
    tokens = _TOKEN_RE.findall(code)
    token_counts = Counter(tokens)
    bigrams = [code[i : i + 2] for i in range(len(code) - 1)]
    bigram_counts = Counter(bigrams)

    return {
        "char_entropy": _entropy(char_counts),
        "token_entropy": _entropy(token_counts),
        "bigram_entropy": _entropy(bigram_counts),
    }


# ------------------------------------------------------------------
# 8. Repetition
# ------------------------------------------------------------------
def _repetition_features(lines: list[str]) -> dict:
    n = max(len(lines), 1)
    stripped = [l.strip() for l in lines if l.strip()]
    counts = Counter(stripped)
    duplicate_count = sum(1 for c in counts.values() if c > 1)

    return {
        "duplicate_line_ratio": duplicate_count / n,
        "unique_line_ratio": len(set(stripped)) / max(len(stripped), 1),
    }


# ======================================================================
# Combine all features for a single code snippet
# ======================================================================
def extract_features(code: str) -> dict:
    """Extract all features from a single code snippet. Returns a flat dict."""
    lines = code.split("\n")
    features = {}
    features.update(_length_features(code, lines))
    features.update(_whitespace_features(lines))
    features.update(_lexical_features(code))
    features.update(_comment_features(code, lines))
    features.update(_structural_features(code, lines))
    features.update(_control_flow_features(code))
    features.update(_punctuation_features(code))
    features.update(_entropy_features(code))
    features.update(_repetition_features(lines))
    return features


# ======================================================================
# Batch extraction (parallel)
# ======================================================================
def _extract_single(code: str) -> dict:
    """Wrapper for multiprocessing (top-level function)."""
    return extract_features(code)


def extract_features_batch(
    codes: list[str] | pd.Series,
    n_workers: int = NUM_WORKERS,
) -> pd.DataFrame:
    """
    Extract features from a list/Series of code snippets in parallel.

    Returns a DataFrame where each row is one snippet and columns are features.
    """
    if isinstance(codes, pd.Series):
        codes = codes.tolist()

    logger.info(f"Extracting features from {len(codes):,} snippets using {n_workers} workers ...")

    with Pool(processes=n_workers) as pool:
        results = pool.map(_extract_single, codes, chunksize=512)

    feature_df = pd.DataFrame(results)
    logger.info(f"Extracted {feature_df.shape[1]} features per snippet.")
    return feature_df


def extract_and_cache(
    df: pd.DataFrame,
    task: str,
    split: str,
    n_workers: int = NUM_WORKERS,
) -> pd.DataFrame:
    """
    Extract features and cache to parquet.  If the cache exists, load from disk.

    Parameters
    ----------
    df : DataFrame with a 'code' column
    task : 'A', 'B', or 'C'
    split : 'train' or 'val'
    """
    cache_path = f"{FEATURES_DIR}/features_{task}_{split}_{len(df)}_{FEATURE_SET_VERSION}.parquet"

    if os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    feature_df = extract_features_batch(df["code"], n_workers=n_workers)
    feature_df.to_parquet(cache_path, index=False)
    logger.info(f"Features cached to {cache_path}")
    return feature_df
