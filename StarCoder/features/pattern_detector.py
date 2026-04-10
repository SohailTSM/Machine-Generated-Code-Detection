"""
Enhanced Pattern Detector
Extracts 57 LLM-specific signature features
Detects patterns unique to different AI code generators
"""

import re
from typing import Dict, List
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnhancedPatternDetector:
    """
    Detect LLM-specific patterns in code.
    
    Features extracted (57 total):
    - GPT family patterns (8)
    - Claude patterns (6)
    - Copilot patterns (5)
    - DeepSeek patterns (6)
    - Llama family patterns (6)
    - StarCoder patterns (5)
    - Gemma patterns (4)
    - Mistral patterns (4)
    - Qwen patterns (5)
    - General AI patterns (8)
    """
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
        self._compile_patterns()
    
    def _get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return [
            # GPT-4/3.5/o patterns (8)
            'gpt_docstring_style', 'gpt_markdown_in_comments', 'gpt_explanatory_comments',
            'gpt_step_by_step', 'gpt_verbose_names', 'gpt_type_everything',
            'gpt_perfect_grammar', 'gpt_educational_tone',
            
            # Claude patterns (6)
            'claude_numbered_steps', 'claude_ultra_verbose_docs', 'claude_polite_comments',
            'claude_careful_error_handling', 'claude_explicit_assumptions', 'claude_formatted_output',
            
            # GitHub Copilot patterns (5)
            'copilot_inline_comments', 'copilot_todo_markers', 'copilot_single_line_funcs',
            'copilot_predictive_naming', 'copilot_context_aware',
            
            # DeepSeek patterns (6)
            'deepseek_fill_style', 'deepseek_heavy_typing', 'deepseek_complete_annotations',
            'deepseek_defensive_checks', 'deepseek_optimal_imports', 'deepseek_clean_structure',
            
            # Llama family patterns (6)
            'llama_long_descriptive_names', 'llama_explicit_logic', 'llama_verbose_conditionals',
            'llama_safety_first', 'llama_conventional_structure', 'llama_clear_separation',
            
            # StarCoder patterns (5)
            'starcoder_typing_module', 'starcoder_star_imports', 'starcoder_idiomatic',
            'starcoder_pattern_matching', 'starcoder_modern_syntax',
            
            # Gemma patterns (4)
            'gemma_balanced_verbosity', 'gemma_practical_comments', 'gemma_efficient_code',
            'gemma_standard_patterns',
            
            # Mistral patterns (4)
            'mistral_concise_style', 'mistral_minimal_comments', 'mistral_direct_approach',
            'mistral_performance_focused',
            
            # Qwen patterns (5)
            'qwen_chinese_influence', 'qwen_thorough_docs', 'qwen_edge_case_handling',
            'qwen_academic_style', 'qwen_comprehensive_tests',
            
            # General AI patterns (8)
            'ai_perfect_formatting', 'ai_consistent_style', 'ai_over_engineering',
            'ai_generic_names', 'ai_complete_error_handling', 'ai_no_dead_code',
            'ai_uniform_complexity', 'ai_standardized_structure'
        ]
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self.patterns = {
            # === GPT PATTERNS ===
            'gpt_docstring_create': re.compile(
                r'"""(?:Generate|Create|Return|Calculate|Process|Handle|Manage|Analyze)', 
                re.IGNORECASE
            ),
            'gpt_markdown': re.compile(r'```\w+|^\s*#\s*\*\*|\*\*\w+\*\*'),
            'gpt_explanatory': re.compile(
                r'#\s*(?:This function|This method|This class|Note that|Important:|First,|Next,|Finally,)',
                re.IGNORECASE
            ),
            'gpt_step': re.compile(r'#\s*Step\s+\d+:', re.IGNORECASE),
            'gpt_verbose_var': re.compile(r'\b[a-z_]{15,}\b'),  # Very long variable names
            'gpt_arrow_return': re.compile(r'->\s*\w+'),
            'gpt_full_sentence': re.compile(r'#\s*[A-Z][^.!?]*[.!?]'),  # Full sentences in comments
            
            # === CLAUDE PATTERNS ===
            'claude_numbered': re.compile(r'#\s*\d+\.|#\s*Step\s+\d+:'),
            'claude_super_long_doc': re.compile(r'"""[\s\S]{300,}"""'),  # Very long docstrings
            'claude_polite': re.compile(
                r'\b(please|kindly|carefully|ensure|make sure|note that)\b',
                re.IGNORECASE
            ),
            'claude_comprehensive_try': re.compile(r'try:[\s\S]{50,}except\s+\w+Error'),
            'claude_assumption': re.compile(r'#\s*(?:Assume|Assuming|Given that)', re.IGNORECASE),
            'claude_formatted_print': re.compile(r'print\(f["\'][\w\s:{}]+["\']\)'),
            
            # === COPILOT PATTERNS ===
            'copilot_inline_freq': re.compile(r'\s+#\s+\w+'),  # Frequent inline comments
            'copilot_todo': re.compile(r'#\s*TODO:', re.IGNORECASE),
            'copilot_one_liner': re.compile(r'^def\s+\w+\([^)]*\):\s*return\s+', re.MULTILINE),
            'copilot_predictive': re.compile(r'\b(result|output|value|data|item|index)\b'),
            'copilot_context': re.compile(r'\b(previous|next|current|last|first)\b'),
            
            # === DEEPSEEK PATTERNS ===
            'deepseek_fill_middle': re.compile(r'<\|fim_.*?\|>'),  # Fill-in-middle markers
            'deepseek_type_everywhere': re.compile(r':\s*(?:int|str|float|bool|list|dict|tuple)\b'),
            'deepseek_full_annotations': re.compile(r'def\s+\w+\([^)]*\)\s*->\s*\w+:'),
            'deepseek_defensive': re.compile(
                r'if\s+(?:not\s+)?\w+\s+is\s+(?:None|not None)|assert\s+\w+'
            ),
            'deepseek_optimal_import': re.compile(r'^from\s+\w+\s+import\s+\w+(?:,\s*\w+)*$', re.MULTILINE),
            
            # === LLAMA PATTERNS ===
            'llama_long_names': re.compile(r'\b[a-z_]{20,}\b'),  # Extremely long names
            'llama_explicit': re.compile(r'#\s*(?:Explicitly|Clearly|Obviously)'),
            'llama_verbose_if': re.compile(r'if\s+[^:]{30,}:'),  # Very long conditions
            'llama_safety': re.compile(
                r'\b(validate|sanitize|check|verify|ensure_safe)\b',
                re.IGNORECASE
            ),
            'llama_conventional': re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']:'),
            
            # === STARCODER PATTERNS ===
            'starcoder_typing': re.compile(r'^from typing import', re.MULTILINE),
            'starcoder_star_import': re.compile(r'from\s+\w+\s+import\s+\*'),
            'starcoder_list_comp': re.compile(r'\[[^\]]+for\s+\w+\s+in\s+[^\]]+\]'),
            'starcoder_match': re.compile(r'\bmatch\s+\w+:'),  # Pattern matching (Python 3.10+)
            'starcoder_walrus': re.compile(r':='),  # Walrus operator
            
            # === GEMMA PATTERNS ===
            'gemma_balanced_comment': re.compile(r'#[^#\n]{10,50}$', re.MULTILINE),
            'gemma_practical': re.compile(r'#\s*(?:Usage|Example|Note):', re.IGNORECASE),
            'gemma_efficient': re.compile(r'\b(cache|optimize|efficient|fast)\b', re.IGNORECASE),
            
            # === MISTRAL PATTERNS ===
            'mistral_minimal_doc': re.compile(r'"""[^"]{10,80}"""'),  # Short docstrings
            'mistral_no_fluff': re.compile(r'^def\s+\w+\([^)]*\):[^#\n]+$', re.MULTILINE),
            'mistral_direct': re.compile(r'\breturn\s+\w+\([^)]*\)'),  # Direct returns
            'mistral_performance': re.compile(r'\b(faster|optimized|efficient|quick)\b', re.IGNORECASE),
            
            # === QWEN PATTERNS ===
            'qwen_pinyin': re.compile(r'\b[a-z]+(?:_[a-z]+){3,}\b'),  # Chinese pinyin style
            'qwen_thorough': re.compile(r'"""[\s\S]{150,300}"""'),  # Medium-long but thorough
            'qwen_edge_cases': re.compile(r'#\s*(?:Edge case|Corner case|Special case)', re.IGNORECASE),
            'qwen_academic': re.compile(r'#\s*(?:Algorithm|Complexity|Reference|Citation)', re.IGNORECASE),
            'qwen_test_comment': re.compile(r'#\s*Test:', re.IGNORECASE),
            
            # === GENERAL AI PATTERNS ===
            'perfect_spacing': re.compile(r'\w+\s*[=+\-*/]\s*\w+'),  # Consistent spacing
            'snake_case_dominant': re.compile(r'\b[a-z]+_[a-z_]+\b'),
            'camel_case_dominant': re.compile(r'\b[a-z]+[A-Z][a-zA-Z]+\b'),
            'generic_var': re.compile(r'\b(result|data|value|item|temp|obj|elem|node)\b'),
            'complete_exception': re.compile(r'except\s+\w+Error(?:\s+as\s+\w+)?:'),
            'no_commented_code': re.compile(r'^\s*#\s*\w+\s*[=\(]', re.MULTILINE),  # Unlikely in AI
            'uniform_indent': re.compile(r'^(?:    )+\w+', re.MULTILINE),  # Consistent indentation
        }
    
    def extract(self, code: str, language: str = 'python') -> List[float]:
        """
        Extract all pattern features from code.
        
        Args:
            code: Source code string
            language: Programming language
            
        Returns:
            List of 57 feature values (normalized 0-1)
        """
        if not code or not isinstance(code, str):
            return [0.0] * len(self.feature_names)
        
        features = {}
        code_len = len(code) + 1
        lines = code.split('\n')
        num_lines = len(lines) + 1
        
        # === GPT PATTERNS (8) ===
        features['gpt_docstring_style'] = (
            len(self.patterns['gpt_docstring_create'].findall(code)) / (code_len / 1000)
        )
        features['gpt_markdown_in_comments'] = (
            len(self.patterns['gpt_markdown'].findall(code)) / (code_len / 1000)
        )
        features['gpt_explanatory_comments'] = (
            len(self.patterns['gpt_explanatory'].findall(code)) / (code_len / 1000)
        )
        features['gpt_step_by_step'] = (
            len(self.patterns['gpt_step'].findall(code)) / (code_len / 1000)
        )
        features['gpt_verbose_names'] = (
            len(self.patterns['gpt_verbose_var'].findall(code)) / (code_len / 1000)
        )
        features['gpt_type_everything'] = (
            len(self.patterns['gpt_arrow_return'].findall(code)) / num_lines
        )
        features['gpt_perfect_grammar'] = (
            len(self.patterns['gpt_full_sentence'].findall(code)) / (code_len / 1000)
        )
        
        # Educational tone (uses "we", "let's", etc.)
        educational = len(re.findall(r'\b(we|let\'s|let us|you can|you should)\b', code, re.IGNORECASE))
        features['gpt_educational_tone'] = educational / (code_len / 1000)
        
        # === CLAUDE PATTERNS (6) ===
        features['claude_numbered_steps'] = (
            len(self.patterns['claude_numbered'].findall(code)) / (code_len / 1000)
        )
        features['claude_ultra_verbose_docs'] = (
            len(self.patterns['claude_super_long_doc'].findall(code)) / (code_len / 5000)
        )
        features['claude_polite_comments'] = (
            len(self.patterns['claude_polite'].findall(code)) / (code_len / 1000)
        )
        features['claude_careful_error_handling'] = (
            len(self.patterns['claude_comprehensive_try'].findall(code)) / (code_len / 2000)
        )
        features['claude_explicit_assumptions'] = (
            len(self.patterns['claude_assumption'].findall(code)) / (code_len / 1000)
        )
        features['claude_formatted_output'] = (
            len(self.patterns['claude_formatted_print'].findall(code)) / num_lines
        )
        
        # === COPILOT PATTERNS (5) ===
        inline_comments = len(self.patterns['copilot_inline_freq'].findall(code))
        features['copilot_inline_comments'] = inline_comments / num_lines
        
        features['copilot_todo_markers'] = (
            len(self.patterns['copilot_todo'].findall(code)) / (code_len / 1000)
        )
        features['copilot_single_line_funcs'] = (
            len(self.patterns['copilot_one_liner'].findall(code)) / (code_len / 1000)
        )
        features['copilot_predictive_naming'] = (
            len(self.patterns['copilot_predictive'].findall(code)) / (code_len / 1000)
        )
        features['copilot_context_aware'] = (
            len(self.patterns['copilot_context'].findall(code)) / (code_len / 1000)
        )
        
        # === DEEPSEEK PATTERNS (6) ===
        features['deepseek_fill_style'] = (
            len(self.patterns['deepseek_fill_middle'].findall(code)) > 0
        ) * 1.0
        
        type_hints = len(self.patterns['deepseek_type_everywhere'].findall(code))
        features['deepseek_heavy_typing'] = type_hints / num_lines
        
        features['deepseek_complete_annotations'] = (
            len(self.patterns['deepseek_full_annotations'].findall(code)) / (code_len / 1000)
        )
        features['deepseek_defensive_checks'] = (
            len(self.patterns['deepseek_defensive'].findall(code)) / num_lines
        )
        features['deepseek_optimal_imports'] = (
            len(self.patterns['deepseek_optimal_import'].findall(code)) / (code_len / 1000)
        )
        
        # Clean structure (no trailing whitespace, consistent newlines)
        clean_lines = sum(1 for line in lines if line == line.rstrip())
        features['deepseek_clean_structure'] = clean_lines / num_lines
        
        # === LLAMA PATTERNS (6) ===
        features['llama_long_descriptive_names'] = (
            len(self.patterns['llama_long_names'].findall(code)) / (code_len / 1000)
        )
        features['llama_explicit_logic'] = (
            len(self.patterns['llama_explicit'].findall(code)) / (code_len / 1000)
        )
        features['llama_verbose_conditionals'] = (
            len(self.patterns['llama_verbose_if'].findall(code)) / num_lines
        )
        features['llama_safety_first'] = (
            len(self.patterns['llama_safety'].findall(code)) / (code_len / 1000)
        )
        features['llama_conventional_structure'] = (
            len(self.patterns['llama_conventional'].findall(code)) > 0
        ) * 1.0
        
        # Clear separation (blank lines between functions)
        blank_lines = sum(1 for line in lines if not line.strip())
        features['llama_clear_separation'] = blank_lines / num_lines
        
        # === STARCODER PATTERNS (5) ===
        features['starcoder_typing_module'] = (
            len(self.patterns['starcoder_typing'].findall(code)) > 0
        ) * 1.0
        features['starcoder_star_imports'] = (
            len(self.patterns['starcoder_star_import'].findall(code)) / (code_len / 1000)
        )
        features['starcoder_idiomatic'] = (
            len(self.patterns['starcoder_list_comp'].findall(code)) / (code_len / 1000)
        )
        features['starcoder_pattern_matching'] = (
            len(self.patterns['starcoder_match'].findall(code)) > 0
        ) * 1.0
        features['starcoder_modern_syntax'] = (
            len(self.patterns['starcoder_walrus'].findall(code)) / (code_len / 1000)
        )
        
        # === GEMMA PATTERNS (4) ===
        features['gemma_balanced_verbosity'] = (
            len(self.patterns['gemma_balanced_comment'].findall(code)) / num_lines
        )
        features['gemma_practical_comments'] = (
            len(self.patterns['gemma_practical'].findall(code)) / (code_len / 1000)
        )
        features['gemma_efficient_code'] = (
            len(self.patterns['gemma_efficient'].findall(code)) / (code_len / 1000)
        )
        
        # Standard patterns (follows PEP 8, common idioms)
        standard_score = 0.5  # Default
        if 'import' in code and '\n\n' in code:  # Has imports and spacing
            standard_score = 0.7
        features['gemma_standard_patterns'] = standard_score
        
        # === MISTRAL PATTERNS (4) ===
        features['mistral_concise_style'] = (
            len(self.patterns['mistral_minimal_doc'].findall(code)) / (code_len / 1000)
        )
        features['mistral_minimal_comments'] = (
            1.0 - (inline_comments / num_lines)  # Inverse of comments
        )
        features['mistral_direct_approach'] = (
            len(self.patterns['mistral_direct'].findall(code)) / (code_len / 1000)
        )
        features['mistral_performance_focused'] = (
            len(self.patterns['mistral_performance'].findall(code)) / (code_len / 1000)
        )
        
        # === QWEN PATTERNS (5) ===
        features['qwen_chinese_influence'] = (
            len(self.patterns['qwen_pinyin'].findall(code)) / (code_len / 1000)
        )
        features['qwen_thorough_docs'] = (
            len(self.patterns['qwen_thorough'].findall(code)) / (code_len / 2000)
        )
        features['qwen_edge_case_handling'] = (
            len(self.patterns['qwen_edge_cases'].findall(code)) / (code_len / 1000)
        )
        features['qwen_academic_style'] = (
            len(self.patterns['qwen_academic'].findall(code)) / (code_len / 1000)
        )
        features['qwen_comprehensive_tests'] = (
            len(self.patterns['qwen_test_comment'].findall(code)) / (code_len / 1000)
        )
        
        # === GENERAL AI PATTERNS (8) ===
        # Perfect formatting
        perfect_spacing = len(self.patterns['perfect_spacing'].findall(code))
        features['ai_perfect_formatting'] = perfect_spacing / (code_len / 1000)
        
        # Consistent style
        snake_count = len(self.patterns['snake_case_dominant'].findall(code))
        camel_count = len(self.patterns['camel_case_dominant'].findall(code))
        total_identifiers = snake_count + camel_count + 1
        features['ai_consistent_style'] = max(snake_count, camel_count) / total_identifiers
        
        # Over-engineering (too many abstractions)
        classes = len(re.findall(r'\bclass\s+\w+', code))
        functions = len(re.findall(r'\bdef\s+\w+', code))
        features['ai_over_engineering'] = min((classes + functions) / num_lines, 1.0)
        
        # Generic names
        features['ai_generic_names'] = (
            len(self.patterns['generic_var'].findall(code)) / (code_len / 1000)
        )
        
        # Complete error handling
        features['ai_complete_error_handling'] = (
            len(self.patterns['complete_exception'].findall(code)) / (code_len / 1000)
        )
        
        # No dead code (commented out code)
        commented_code = len(self.patterns['no_commented_code'].findall(code))
        features['ai_no_dead_code'] = 1.0 - min(commented_code / num_lines, 1.0)
        
        # Uniform complexity (similar-length functions)
        function_matches = re.finditer(r'^def\s+\w+', code, re.MULTILINE)
        func_positions = [m.start() for m in function_matches]
        
        if len(func_positions) > 1:
            func_lengths = []
            for i in range(len(func_positions) - 1):
                length = func_positions[i + 1] - func_positions[i]
                func_lengths.append(length)
            
            # Low std = uniform
            features['ai_uniform_complexity'] = 1.0 - min(np.std(func_lengths) / 1000, 1.0)
        else:
            features['ai_uniform_complexity'] = 0.5
        
        # Standardized structure
        uniform_indent = len(self.patterns['uniform_indent'].findall(code))
        features['ai_standardized_structure'] = uniform_indent / num_lines
        
        # Normalize all features to [0, 1] and clip
        for key in features:
            features[key] = min(max(features[key], 0.0), 1.0)
        
        # Return in consistent order
        return [features[name] for name in self.feature_names]
    
    def get_generator_scores(self, features: List[float]) -> Dict[str, float]:
        """
        Aggregate features into per-generator scores.
        
        Args:
            features: List of 57 feature values
            
        Returns:
            Dict mapping generator name to score
        """
        feat_dict = dict(zip(self.feature_names, features))
        
        scores = {
            'gpt': np.mean([feat_dict[f] for f in feat_dict if f.startswith('gpt_')]),
            'claude': np.mean([feat_dict[f] for f in feat_dict if f.startswith('claude_')]),
            'copilot': np.mean([feat_dict[f] for f in feat_dict if f.startswith('copilot_')]),
            'deepseek': np.mean([feat_dict[f] for f in feat_dict if f.startswith('deepseek_')]),
            'llama': np.mean([feat_dict[f] for f in feat_dict if f.startswith('llama_')]),
            'starcoder': np.mean([feat_dict[f] for f in feat_dict if f.startswith('starcoder_')]),
            'gemma': np.mean([feat_dict[f] for f in feat_dict if f.startswith('gemma_')]),
            'mistral': np.mean([feat_dict[f] for f in feat_dict if f.startswith('mistral_')]),
            'qwen': np.mean([feat_dict[f] for f in feat_dict if f.startswith('qwen_')]),
            'ai_generic': np.mean([feat_dict[f] for f in feat_dict if f.startswith('ai_')])
        }
        
        return scores


if __name__ == "__main__":
    # Test the detector
    detector = EnhancedPatternDetector()
    
    test_code = '''
def calculate_fibonacci_sequence(n: int) -> list:
    """
    Generate a Fibonacci sequence up to n numbers.
    
    This function creates a list containing the Fibonacci sequence.
    The Fibonacci sequence is a series where each number is the sum
    of the two preceding ones.
    
    Args:
        n: The number of Fibonacci numbers to generate
        
    Returns:
        A list containing the Fibonacci sequence
    """
    # Step 1: Initialize the sequence with first two numbers
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    # Step 2: Build the sequence
    fibonacci_sequence = [0, 1]
    
    # Step 3: Generate remaining numbers
    for index in range(2, n):
        next_number = fibonacci_sequence[index - 1] + fibonacci_sequence[index - 2]
        fibonacci_sequence.append(next_number)
    
    return fibonacci_sequence
'''
    
    features = detector.extract(test_code)
    print(f"Extracted {len(features)} pattern features")
    
    scores = detector.get_generator_scores(features)
    print("\nGenerator scores:")
    for gen, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {gen}: {score:.4f}")
