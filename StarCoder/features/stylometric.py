"""
Stylometric Feature Extractor

Extracts 12 code style and formatting features that distinguish human from AI code:
1. Average identifier length
2. Identifier naming entropy
3. Naming consistency
4. Whitespace pattern entropy
5. Indentation consistency
6. Comment density
7. Comment style variance
8. Line length variance
9. Token diversity (unique tokens / total tokens)
10. Human markers (TODO, FIXME, HACK, etc.)
11. Code-to-comment ratio
12. Formatting consistency

AI-generated code typically has:
- More consistent formatting
- Longer, more descriptive identifiers
- Higher comment density
- Lower style variance
- Fewer human markers (TODO, FIXME, etc.)
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Set
import tokenize
import io


class StylometricExtractor:
    """Extract code style and formatting features."""
    
    # Human markers that appear more in human code
    HUMAN_MARKERS = [
        'TODO', 'FIXME', 'HACK', 'XXX', 'NOTE', 'BUG',
        'OPTIMIZE', 'REFACTOR', 'DEPRECATE', 'TEMP'
    ]
    
    # Comment patterns
    SINGLE_LINE_COMMENT = re.compile(r'#.*$', re.MULTILINE)
    DOCSTRING_SINGLE = re.compile(r'""".*?"""', re.DOTALL)
    DOCSTRING_DOUBLE = re.compile(r"'''.*?'''", re.DOTALL)
    
    # Identifier patterns
    IDENTIFIER_PATTERN = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    CAMEL_CASE = re.compile(r'[a-z][A-Z]')
    SNAKE_CASE = re.compile(r'[a-z]_[a-z]')
    
    def __init__(self):
        pass
    
    def extract_identifiers(self, code: str) -> List[str]:
        """Extract all identifiers from code."""
        # Remove strings and comments to avoid false identifiers
        code_cleaned = re.sub(r'["\'].*?["\']', '', code)
        code_cleaned = re.sub(r'#.*$', '', code_cleaned, flags=re.MULTILINE)
        
        identifiers = self.IDENTIFIER_PATTERN.findall(code_cleaned)
        
        # Filter out keywords
        keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
            'while', 'with', 'yield'
        }
        
        return [ident for ident in identifiers if ident not in keywords]
    
    def compute_identifier_length(self, identifiers: List[str]) -> float:
        """Average identifier length."""
        if not identifiers:
            return 0.0
        return np.mean([len(ident) for ident in identifiers])
    
    def compute_identifier_entropy(self, identifiers: List[str]) -> float:
        """
        Entropy of identifier lengths.
        Higher entropy = more varied naming.
        """
        if not identifiers:
            return 0.0
        
        lengths = [len(ident) for ident in identifiers]
        counter = Counter(lengths)
        total = sum(counter.values())
        
        entropy = 0.0
        for count in counter.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def compute_naming_consistency(self, identifiers: List[str]) -> float:
        """
        Measure consistency of naming conventions.
        
        Returns score [0-1] where 1 = perfectly consistent.
        AI code tends to be more consistent.
        """
        if not identifiers:
            return 0.0
        
        camel_case = sum(1 for ident in identifiers if self.CAMEL_CASE.search(ident))
        snake_case = sum(1 for ident in identifiers if self.SNAKE_CASE.search(ident))
        
        total = len(identifiers)
        
        # Consistency = max convention usage / total
        consistency = max(camel_case, snake_case) / total if total > 0 else 0.0
        
        return consistency
    
    def compute_whitespace_entropy(self, code: str) -> float:
        """
        Entropy of whitespace patterns.
        
        AI code tends to have lower entropy (more consistent spacing).
        """
        # Extract whitespace sequences
        whitespace_pattern = re.compile(r'\s+')
        whitespaces = whitespace_pattern.findall(code)
        
        if not whitespaces:
            return 0.0
        
        # Count different whitespace types
        counter = Counter(whitespaces)
        total = sum(counter.values())
        
        entropy = 0.0
        for count in counter.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        # Normalize to [0, 1]
        # Max entropy for whitespace is typically ~4 bits
        return min(1.0, entropy / 4.0)
    
    def compute_indentation_consistency(self, code: str) -> float:
        """
        Measure indentation consistency.
        
        Returns score [0-1] where 1 = perfectly consistent.
        AI code is typically more consistent.
        """
        lines = code.split('\n')
        indents = []
        
        for line in lines:
            if line.strip():  # Non-empty lines
                # Count leading whitespace
                indent = len(line) - len(line.lstrip())
                indents.append(indent)
        
        if not indents:
            return 0.0
        
        # Check if all indents are multiples of the same base
        indents = np.array(indents)
        if len(indents) < 2:
            return 1.0
        
        # Find most common non-zero indent step
        diffs = np.diff(sorted(set(indents)))
        diffs = diffs[diffs > 0]
        
        if len(diffs) == 0:
            return 1.0
        
        # Most common step should be GCD of all indents
        from math import gcd
        from functools import reduce
        
        indent_gcd = reduce(gcd, [i for i in indents if i > 0])
        
        if indent_gcd == 0:
            return 1.0
        
        # Check if all indents are multiples
        consistency = sum(1 for i in indents if i % indent_gcd == 0) / len(indents)
        
        return consistency
    
    def compute_comment_density(self, code: str) -> float:
        """
        Ratio of comment lines to total lines.
        
        AI code tends to have higher comment density.
        """
        lines = code.split('\n')
        total_lines = len([l for l in lines if l.strip()])
        
        if total_lines == 0:
            return 0.0
        
        # Count comment lines
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        
        # Count docstring lines
        docstrings = self.DOCSTRING_SINGLE.findall(code) + self.DOCSTRING_DOUBLE.findall(code)
        docstring_lines = sum(len(ds.split('\n')) for ds in docstrings)
        
        total_comment_lines = comment_lines + docstring_lines
        
        return min(1.0, total_comment_lines / total_lines)
    
    def compute_comment_style_variance(self, code: str) -> float:
        """
        Variance in comment styles.
        
        AI code tends to have more uniform comment style.
        Returns variance [0-1] where lower = more consistent (AI-like).
        """
        comments = self.SINGLE_LINE_COMMENT.findall(code)
        
        if not comments:
            return 0.0
        
        # Analyze comment lengths
        lengths = [len(c.strip('#').strip()) for c in comments]
        
        if len(lengths) < 2:
            return 0.0
        
        # Compute coefficient of variation
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        if mean_len == 0:
            return 0.0
        
        cv = std_len / mean_len
        
        # Normalize to [0, 1]
        return min(1.0, cv)
    
    def compute_line_length_variance(self, code: str) -> float:
        """
        Variance in line lengths.
        
        AI code tends to have more consistent line lengths.
        """
        lines = [l for l in code.split('\n') if l.strip()]
        
        if not lines:
            return 0.0
        
        lengths = [len(l) for l in lines]
        
        if len(lengths) < 2:
            return 0.0
        
        # Compute coefficient of variation
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        if mean_len == 0:
            return 0.0
        
        cv = std_len / mean_len
        
        # Normalize to [0, 1]
        return min(1.0, cv)
    
    def compute_token_diversity(self, code: str) -> float:
        """
        Unique tokens / total tokens.
        
        AI code may have lower diversity (more repetitive patterns).
        """
        try:
            tokens = []
            readline = io.BytesIO(code.encode('utf-8')).readline
            
            for tok in tokenize.tokenize(readline):
                if tok.type != tokenize.ENCODING and tok.type != tokenize.ENDMARKER:
                    tokens.append(tok.string)
            
            if not tokens:
                return 0.0
            
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            
            return unique_tokens / total_tokens
            
        except:
            # Fallback: simple whitespace tokenization
            tokens = code.split()
            if not tokens:
                return 0.0
            return len(set(tokens)) / len(tokens)
    
    def compute_human_markers(self, code: str) -> float:
        """
        Count of human markers (TODO, FIXME, etc.).
        
        Higher count suggests human code.
        Returns normalized count [0-1].
        """
        code_upper = code.upper()
        
        marker_count = sum(code_upper.count(marker) for marker in self.HUMAN_MARKERS)
        
        # Normalize by number of lines
        lines = len([l for l in code.split('\n') if l.strip()])
        if lines == 0:
            return 0.0
        
        # Typical human code has 0-0.1 markers per line
        normalized = min(1.0, marker_count / (lines * 0.1))
        
        return normalized
    
    def compute_code_to_comment_ratio(self, code: str) -> float:
        """
        Ratio of code lines to comment lines.
        
        AI code tends to have higher comment-to-code ratio (lower this value).
        """
        lines = code.split('\n')
        
        code_lines = 0
        comment_lines = 0
        
        in_docstring = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for docstring
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                comment_lines += 1
            elif in_docstring:
                comment_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            elif stripped:
                code_lines += 1
        
        if comment_lines == 0:
            return 1.0  # All code, no comments
        
        ratio = code_lines / comment_lines
        
        # Normalize to [0, 1]: typical ratio is 1-20
        # Lower ratio (more comments) = AI-like
        return 1.0 - min(1.0, ratio / 20.0)
    
    def compute_formatting_consistency(self, code: str) -> float:
        """
        Overall formatting consistency score.
        
        Combines multiple formatting metrics.
        AI code is typically more consistent.
        """
        # Check consistency of:
        # 1. Blank lines between functions
        # 2. Spacing around operators
        # 3. Quote style consistency
        
        consistency_score = 0.0
        checks = 0
        
        # Check 1: Blank lines before function definitions
        func_pattern = re.compile(r'\n\s*def\s+\w+')
        func_matches = list(func_pattern.finditer(code))
        
        if len(func_matches) > 1:
            blank_lines_before = []
            for match in func_matches[1:]:  # Skip first function
                before_func = code[:match.start()].split('\n')
                # Count trailing blank lines
                blank_count = 0
                for line in reversed(before_func):
                    if line.strip():
                        break
                    blank_count += 1
                blank_lines_before.append(blank_count)
            
            if blank_lines_before:
                # Check if consistent (2 blank lines is PEP8 standard)
                consistency_score += 1.0 - (np.std(blank_lines_before) / 3.0)
                checks += 1
        
        # Check 2: Spacing around operators
        operators = [' = ', ' == ', ' + ', ' - ', ' * ', ' / ']
        spaced_ops = sum(code.count(op) for op in operators)
        
        unspaced_ops = sum(code.count(op.strip()) for op in operators)
        
        if unspaced_ops > 0:
            spacing_consistency = spaced_ops / unspaced_ops
            consistency_score += min(1.0, spacing_consistency)
            checks += 1
        
        # Check 3: Quote style consistency
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        
        total_quotes = single_quotes + double_quotes
        if total_quotes > 0:
            quote_consistency = max(single_quotes, double_quotes) / total_quotes
            consistency_score += quote_consistency
            checks += 1
        
        if checks == 0:
            return 0.5  # Neutral score
        
        return consistency_score / checks
    
    def extract(self, code: str, language: str = "python") -> List[float]:
        """
        Extract all 12 stylometric features.
        
        Features:
        1. Average identifier length
        2. Identifier naming entropy
        3. Naming consistency
        4. Whitespace pattern entropy
        5. Indentation consistency
        6. Comment density
        7. Comment style variance
        8. Line length variance
        9. Token diversity
        10. Human markers count
        11. Code-to-comment ratio
        12. Formatting consistency
        
        Returns:
            List of 12 normalized features [0-1]
        """
        if not code or len(code.strip()) == 0:
            return [0.0] * 12
        
        features = []
        
        try:
            # Extract identifiers once for multiple features
            identifiers = self.extract_identifiers(code)
            
            # Feature 1: Average identifier length
            avg_length = self.compute_identifier_length(identifiers)
            features.append(min(1.0, avg_length / 20.0))  # Normalize: typical 5-20 chars
            
            # Feature 2: Identifier entropy
            ident_entropy = self.compute_identifier_entropy(identifiers)
            features.append(min(1.0, ident_entropy / 5.0))  # Normalize: typical 0-5 bits
            
            # Feature 3: Naming consistency
            naming_consistency = self.compute_naming_consistency(identifiers)
            features.append(naming_consistency)
            
            # Feature 4: Whitespace entropy
            whitespace_entropy = self.compute_whitespace_entropy(code)
            features.append(whitespace_entropy)
            
            # Feature 5: Indentation consistency
            indent_consistency = self.compute_indentation_consistency(code)
            features.append(indent_consistency)
            
            # Feature 6: Comment density
            comment_density = self.compute_comment_density(code)
            features.append(comment_density)
            
            # Feature 7: Comment style variance
            comment_variance = self.compute_comment_style_variance(code)
            features.append(comment_variance)
            
            # Feature 8: Line length variance
            line_variance = self.compute_line_length_variance(code)
            features.append(line_variance)
            
            # Feature 9: Token diversity
            token_diversity = self.compute_token_diversity(code)
            features.append(token_diversity)
            
            # Feature 10: Human markers
            human_markers = self.compute_human_markers(code)
            features.append(human_markers)
            
            # Feature 11: Code-to-comment ratio
            code_comment_ratio = self.compute_code_to_comment_ratio(code)
            features.append(code_comment_ratio)
            
            # Feature 12: Formatting consistency
            formatting = self.compute_formatting_consistency(code)
            features.append(formatting)
            
        except Exception as e:
            print(f"Error extracting stylometric features: {e}")
            features = [0.0] * 12
        
        # Ensure we have exactly 12 features
        return features[:12] + [0.0] * max(0, 12 - len(features))


if __name__ == "__main__":
    # Test the stylometric extractor
    print("Testing Stylometric Feature Extractor")
    print("=" * 50)
    
    # Sample codes
    human_code = """
def calc(x,y):
    #TODO: add error handling
    result=x+y
    return result

def process_data(data):
    # FIXME: this is slow
    output=[]
    for item in data:
        output.append(calc(item,10))
    return output
"""
    
    ai_code = """
def calculate_sum(number_one: int, number_two: int) -> int:
    '''
    Calculate the sum of two integers with proper error handling.
    
    This function takes two integer parameters and returns their sum.
    It ensures type safety through type hints and provides comprehensive
    documentation for maintainability.
    
    Args:
        number_one: The first integer to be added
        number_two: The second integer to be added
        
    Returns:
        The sum of the two input integers
        
    Example:
        >>> calculate_sum(5, 3)
        8
    '''
    result_value = number_one + number_two
    return result_value


def process_data_items(data_collection: List[int]) -> List[int]:
    '''
    Process a collection of data items by applying calculation.
    
    This function iterates through each item in the input collection
    and applies the calculate_sum function with a constant value of 10.
    
    Args:
        data_collection: A list of integers to be processed
        
    Returns:
        A list of processed integer values
    '''
    output_results = []
    for individual_item in data_collection:
        processed_value = calculate_sum(individual_item, 10)
        output_results.append(processed_value)
    return output_results
"""
    
    print("\nInitializing extractor...")
    extractor = StylometricExtractor()
    
    print("\nExtracting features from human code:")
    human_features = extractor.extract(human_code)
    print(f"Features: {[f'{f:.3f}' for f in human_features]}")
    
    print("\nExtracting features from AI code:")
    ai_features = extractor.extract(ai_code)
    print(f"Features: {[f'{f:.3f}' for f in ai_features]}")
    
    print("\nFeature comparison (Human vs AI):")
    feature_names = [
        "Avg Ident Length",
        "Ident Entropy",
        "Naming Consist",
        "Whitespace Ent",
        "Indent Consist",
        "Comment Density",
        "Comment Variance",
        "Line Len Var",
        "Token Diversity",
        "Human Markers",
        "Code/Comment Ratio",
        "Format Consist"
    ]
    
    for name, h_feat, a_feat in zip(feature_names, human_features, ai_features):
        diff = a_feat - h_feat
        print(f"  {name:20s}: Human={h_feat:.3f}, AI={a_feat:.3f}, Diff={diff:+.3f}")
    
    print("\n" + "=" * 50)
    print("Expected AI patterns:")
    print("  - Longer identifiers")
    print("  - Higher naming consistency")
    print("  - Higher indentation consistency")
    print("  - Higher comment density")
    print("  - Lower comment variance")
    print("  - Fewer human markers (TODO, FIXME)")
    print("  - Higher formatting consistency")
