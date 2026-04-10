"""Tree-Sitter parsing — parser registry, parse_code(), language detection."""

import importlib
from collections import Counter

from tree_sitter import Language, Parser

# ── Language name mapping ──
_LANG_MAP = {
    "python": "python", "py": "python",
    "c++": "cpp", "cpp": "cpp",
    "java": "java",
    "javascript": "javascript", "js": "javascript",
    "go": "go", "golang": "go",
    "c#": "c_sharp", "csharp": "c_sharp", "c_sharp": "c_sharp",
    "c": "c",
    "php": "php",
}

_UNKNOWN_LANG_COUNTER = Counter()
_parsers = {}


def _make_parser(lang_id: str):
    """Instantiate a Tree-sitter Parser for the given language id."""
    try:
        mod = importlib.import_module(f"tree_sitter_{lang_id}")
        if lang_id == "php":
            fn = getattr(mod, "language_php", None) or getattr(mod, "language", None)
            if fn is None:
                raise RuntimeError("PHP grammar entrypoint not found")
            lang_capsule = fn()
        else:
            lang_capsule = mod.language()
        lang = Language(lang_capsule)
        parser = Parser()
        parser.language = lang
        return parser
    except ModuleNotFoundError:
        print(f"  ⚠ tree_sitter_{lang_id} not installed — skipping.")
        return None
    except Exception as exc:
        print(f"  ⚠ Could not init parser for {lang_id}: {exc}")
        return None


def init_parsers():
    """Build parser registry. Call once at startup."""
    global _parsers
    if _parsers:
        return _parsers
    for lang_id in sorted(set(_LANG_MAP.values())):
        p = _make_parser(lang_id)
        if p is not None:
            _parsers[lang_id] = p
    if not _parsers:
        raise RuntimeError("FATAL: No Tree-sitter parsers could be loaded.")
    print(f"✓ Loaded parsers for: {sorted(_parsers.keys())}")
    return _parsers


def get_parsers():
    """Get the parser registry, initializing if needed."""
    if not _parsers:
        init_parsers()
    return _parsers


def parse_code(code_str: str, lang_hint: str):
    """Parse code string → AST root node. Returns None on failure.

    Does NOT fall back to Python for unknown languages (v5 fix).
    """
    if not code_str or not code_str.strip():
        return None
    try:
        parsers = get_parsers()
        mapped = _LANG_MAP.get(str(lang_hint).lower().strip())
        if mapped is None or mapped not in parsers:
            _UNKNOWN_LANG_COUNTER[str(lang_hint)] += 1
            return None
        tree = parsers[mapped].parse(code_str.encode("utf-8", errors="replace"))
        return tree.root_node
    except Exception:
        return None


# ── Language detection for hidden test set ──
_DETECT_MAP = {
    "#include":      lambda c: "C++" if ("class" in c or "cout" in c or "::" in c) else "C",
    "System.out":    lambda _: "Java",
    "public class":  lambda _: "Java",
    "<?php":         lambda _: "PHP",
    "using System":  lambda _: "C#",
    "func main()":   lambda _: "Go",
    "package main":  lambda _: "Go",
    "console.log":   lambda _: "JavaScript",
    "function ":     lambda _: "JavaScript",
}


def detect_language_fast(code: str) -> str:
    """Heuristic language detection for test set (no language column)."""
    for keyword, resolver in _DETECT_MAP.items():
        if keyword in code:
            return resolver(code)
    return "Python"


def get_unknown_lang_stats():
    """Return counter of unknown languages encountered during parsing."""
    return dict(_UNKNOWN_LANG_COUNTER)
