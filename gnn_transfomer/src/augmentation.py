"""Adversarial data augmentation — rename, reformat, add comments."""

import random
import re

from src.tree_sitter_utils import parse_code


class CodeAugmenter:
    """Three augmentation types for training robustness.

    1. Rename identifiers → var_0, var_1, ...
    2. Reformat whitespace (indent changes, blank lines)
    3. Add dummy comments
    """

    def __init__(self, rename_prob=0.5, reformat_prob=0.3, comment_prob=0.2):
        self.rename_prob = rename_prob
        self.reformat_prob = reformat_prob
        self.comment_prob = comment_prob

        self._dummy_comments = [
            "# Process data", "# Initialize", "# Check condition",
            "# Update state", "# Helper function", "# Main logic",
            "// Process data", "// Initialize", "// Check condition",
            "// Update state", "// Helper function", "// Main logic",
        ]

    def augment(self, code: str, language: str = "Python") -> str:
        """Apply 1-2 random augmentations. Returns augmented code string."""
        augmentations = []
        if random.random() < self.rename_prob:
            augmentations.append(self._rename_identifiers)
        if random.random() < self.reformat_prob:
            augmentations.append(self._reformat_whitespace)
        if random.random() < self.comment_prob:
            augmentations.append(self._add_dummy_comments)

        if not augmentations:
            augmentations.append(random.choice([
                self._rename_identifiers,
                self._reformat_whitespace,
                self._add_dummy_comments,
            ]))

        result = code
        for aug_fn in augmentations:
            try:
                result = aug_fn(result, language)
            except Exception:
                pass  # Gracefully skip failed augmentations
        return result

    def _rename_identifiers(self, code: str, language: str) -> str:
        """Parse AST, find identifiers, replace with var_0, var_1, etc."""
        root = parse_code(code, language)
        if root is None:
            return code

        # Collect all identifier nodes
        identifiers = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "identifier" and node.text:
                text = node.text.decode("utf-8", errors="replace")
                # Don't rename very short or built-in names
                if len(text) > 1 and text not in {
                    "self", "this", "cls", "None", "True", "False",
                    "null", "true", "false", "main", "args",
                    "int", "str", "float", "bool", "list", "dict",
                    "print", "return", "import", "from", "def", "class",
                }:
                    identifiers.add(text)
            stack.extend(node.children)

        if not identifiers:
            return code

        # Create rename mapping
        rename_map = {}
        for i, name in enumerate(sorted(identifiers)):
            rename_map[name] = f"var_{i}"

        # Apply renaming (word-boundary aware)
        result = code
        for old_name, new_name in sorted(rename_map.items(),
                                          key=lambda x: -len(x[0])):
            result = re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, result)
        return result

    def _reformat_whitespace(self, code: str, language: str = "") -> str:
        """Randomly alter indentation and blank lines."""
        lines = code.split("\n")
        result = []
        for line in lines:
            # Random blank line insertion (~5% chance)
            if random.random() < 0.05:
                result.append("")

            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if stripped and indent > 0:
                # Randomly double or halve indentation
                r = random.random()
                if r < 0.3:
                    indent = indent * 2
                elif r < 0.5:
                    indent = max(1, indent // 2)

            result.append(" " * indent + stripped)

            # Random blank line removal (skip next empty)
            if random.random() < 0.05 and result and result[-1] == "":
                result.pop()

        return "\n".join(result)

    def _add_dummy_comments(self, code: str, language: str = "") -> str:
        """Insert benign comments at random positions."""
        lines = code.split("\n")
        result = []
        for line in lines:
            result.append(line)
            if random.random() < 0.08:  # ~8% chance per line
                indent = len(line) - len(line.lstrip())
                comment = random.choice(self._dummy_comments)
                result.append(" " * indent + comment)
        return "\n".join(result)
