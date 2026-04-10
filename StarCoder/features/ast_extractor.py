"""
Enhanced AST Feature Extractor
Extracts 33 comprehensive features from Abstract Syntax Trees
Supports multiple languages with fallback for unsupported languages
"""

import ast
import re
import numpy as np
from typing import Dict, List, Optional, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class EnhancedASTExtractor:
    """
    Extract comprehensive structural features from code AST.
    
    Features extracted (33 total):
    - Tree structure: depth, breadth, balance
    - Node statistics: type distribution, entropy
    - Control flow complexity: cyclomatic, cognitive
    - Function/class metrics
    - Naming patterns
    - Documentation patterns
    - Type hint usage
    - Exception handling patterns
    - Import patterns
    - Code organization metrics
    """
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
        self.supported_languages = ['python']  # Extensible
    
    def _get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return [
            # Tree structure (5)
            'max_depth', 'avg_depth', 'depth_std', 'max_breadth', 'tree_balance',
            
            # Node statistics (3)
            'node_entropy', 'unique_node_ratio', 'total_nodes',
            
            # Control flow (5)
            'if_count', 'loop_count', 'try_count', 'cyclomatic_complexity', 'nesting_depth',
            
            # Function metrics (6)
            'num_functions', 'avg_function_length', 'max_function_length',
            'avg_function_params', 'max_function_params', 'avg_function_complexity',
            
            # Class metrics (3)
            'num_classes', 'avg_methods_per_class', 'inheritance_depth',
            
            # Documentation (3)
            'docstring_ratio', 'comment_density', 'avg_docstring_length',
            
            # Type hints (2)
            'type_hint_ratio', 'return_annotation_ratio',
            
            # Exception handling (2)
            'exception_specificity', 'try_coverage',
            
            # Naming patterns (2)
            'naming_consistency', 'avg_identifier_length',
            
            # Code organization (2)
            'import_count', 'import_diversity'
        ]
    
    def extract(self, code: str, language: str = 'python') -> List[float]:
        """
        Extract AST features from code.
        
        Args:
            code: Source code string
            language: Programming language
            
        Returns:
            List of 33 feature values
        """
        if language == 'python':
            return self._extract_python_features(code)
        else:
            return self._extract_fallback_features(code)
    
    def _extract_python_features(self, code: str) -> List[float]:
        """Extract features from Python AST."""
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError) as e:
            logger.debug(f"AST parse failed: {e}")
            return self._default_features()
        
        features = {}
        
        # === TREE STRUCTURE ===
        depths = []
        breadths = {}  # depth -> count of nodes at that depth
        
        def traverse_tree(node, depth=0):
            depths.append(depth)
            breadths[depth] = breadths.get(depth, 0) + 1
            for child in ast.iter_child_nodes(node):
                traverse_tree(child, depth + 1)
        
        traverse_tree(tree)
        
        features['max_depth'] = max(depths) if depths else 0
        features['avg_depth'] = np.mean(depths) if depths else 0
        features['depth_std'] = np.std(depths) if len(depths) > 1 else 0
        features['max_breadth'] = max(breadths.values()) if breadths else 0
        
        # Tree balance (ratio of avg_depth to max_depth)
        features['tree_balance'] = (
            features['avg_depth'] / features['max_depth']
            if features['max_depth'] > 0 else 0
        )
        
        # === NODE STATISTICS ===
        all_nodes = list(ast.walk(tree))
        node_types = [type(node).__name__ for node in all_nodes]
        type_counts = Counter(node_types)
        
        features['total_nodes'] = len(all_nodes)
        features['unique_node_ratio'] = len(type_counts) / len(all_nodes) if all_nodes else 0
        
        # Node type entropy
        total = len(node_types)
        if total > 0:
            probs = [c / total for c in type_counts.values()]
            features['node_entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            features['node_entropy'] = 0
        
        # === CONTROL FLOW ===
        features['if_count'] = sum(1 for n in all_nodes if isinstance(n, ast.If))
        features['loop_count'] = sum(1 for n in all_nodes 
                                     if isinstance(n, (ast.For, ast.While)))
        features['try_count'] = sum(1 for n in all_nodes if isinstance(n, ast.Try))
        
        # Cyclomatic complexity (McCabe)
        features['cyclomatic_complexity'] = self._calculate_cyclomatic(tree)
        
        # Maximum nesting depth
        features['nesting_depth'] = self._calculate_nesting_depth(tree)
        
        # === FUNCTION METRICS ===
        functions = [n for n in all_nodes if isinstance(n, ast.FunctionDef)]
        features['num_functions'] = len(functions)
        
        if functions:
            func_lengths = []
            func_params = []
            func_complexities = []
            
            for func in functions:
                # Length (number of lines)
                if hasattr(func, 'end_lineno') and hasattr(func, 'lineno'):
                    func_lengths.append(func.end_lineno - func.lineno + 1)
                
                # Parameters
                func_params.append(len(func.args.args))
                
                # Complexity (branches within function)
                func_nodes = list(ast.walk(func))
                branches = sum(1 for n in func_nodes 
                              if isinstance(n, (ast.If, ast.For, ast.While, ast.With)))
                func_complexities.append(branches)
            
            features['avg_function_length'] = np.mean(func_lengths) if func_lengths else 0
            features['max_function_length'] = max(func_lengths) if func_lengths else 0
            features['avg_function_params'] = np.mean(func_params)
            features['max_function_params'] = max(func_params) if func_params else 0
            features['avg_function_complexity'] = np.mean(func_complexities)
        else:
            features['avg_function_length'] = 0
            features['max_function_length'] = 0
            features['avg_function_params'] = 0
            features['max_function_params'] = 0
            features['avg_function_complexity'] = 0
        
        # === CLASS METRICS ===
        classes = [n for n in all_nodes if isinstance(n, ast.ClassDef)]
        features['num_classes'] = len(classes)
        
        if classes:
            methods_per_class = []
            for cls in classes:
                methods = [n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
                methods_per_class.append(len(methods))
            
            features['avg_methods_per_class'] = np.mean(methods_per_class)
            
            # Inheritance depth (bases count)
            inheritance = [len(cls.bases) for cls in classes]
            features['inheritance_depth'] = max(inheritance) if inheritance else 0
        else:
            features['avg_methods_per_class'] = 0
            features['inheritance_depth'] = 0
       
        # === DOCUMENTATION ===
        docstring_count = 0
        docstring_lengths = []
        
        for node in functions + classes:
            docstring = ast.get_docstring(node)
            if docstring:
                docstring_count += 1
                docstring_lengths.append(len(docstring))
        
        total_documentable = len(functions) + len(classes)
        features['docstring_ratio'] = (
            docstring_count / total_documentable if total_documentable > 0 else 0
        )
        features['avg_docstring_length'] = (
            np.mean(docstring_lengths) if docstring_lengths else 0
        )
        
        # Comment density (from source)
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        features['comment_density'] = comment_lines / len(lines) if lines else 0
        
        # === TYPE HINTS ===
        # Argument type hints
        total_args = 0
        typed_args = 0
        
        for func in functions:
            for arg in func.args.args:
                total_args += 1
                if arg.annotation is not None:
                    typed_args += 1
        
        features['type_hint_ratio'] = typed_args / total_args if total_args > 0 else 0
        
        # Return annotations
        annotated_returns = sum(1 for func in functions if func.returns is not None)
        features['return_annotation_ratio'] = (
            annotated_returns / len(functions) if functions else 0
        )
        
        # === EXCEPTION HANDLING ===
        exceptions = [n for n in all_nodes if isinstance(n, ast.ExceptHandler)]
        
        # Specificity (specific exception vs bare except)
        specific_exceptions = sum(1 for exc in exceptions if exc.type is not None)
        features['exception_specificity'] = (
            specific_exceptions / len(exceptions) if exceptions else 0.5
        )
        
        # Try coverage (ratio of code in try blocks)
        try_nodes = [n for n in all_nodes if isinstance(n, ast.Try)]
        try_statement_count = sum(len(list(ast.walk(t))) for t in try_nodes)
        features['try_coverage'] = (
            try_statement_count / len(all_nodes) if all_nodes else 0
        )
        
        # === NAMING PATTERNS ===
        names = [n.id for n in all_nodes if isinstance(n, ast.Name)]
        
        if names:
            # Consistency (snake_case vs camelCase)
            snake_case = sum(1 for n in names if '_' in n and n.islower())
            camel_case = sum(1 for n in names if any(c.isupper() for c in n) and '_' not in n)
            
            # AI tends to be more consistent
            features['naming_consistency'] = max(snake_case, camel_case) / len(names)
            
            # Average identifier length
            features['avg_identifier_length'] = np.mean([len(n) for n in names])
        else:
            features['naming_consistency'] = 0
            features['avg_identifier_length'] = 0
        
        # === CODE ORGANIZATION ===
        # Import statements
        imports = [n for n in all_nodes if isinstance(n, (ast.Import, ast.ImportFrom))]
        features['import_count'] = len(imports)
        
        # Import diversity (unique modules)
        imported_modules = set()
        for imp in imports:
            if isinstance(imp, ast.Import):
                imported_modules.update(alias.name for alias in imp.names)
            elif isinstance(imp, ast.ImportFrom) and imp.module:
                imported_modules.add(imp.module)
        
        features['import_diversity'] = (
            len(imported_modules) / features['import_count']
            if features['import_count'] > 0 else 0
        )
        
        # Return features in consistent order
        return [features[name] for name in self.feature_names]
    
    def _calculate_cyclomatic(self, tree: ast.AST) -> int:
        """Calculate McCabe cyclomatic complexity."""
        # M = E - N + 2P
        # For simplified calculation: count decision points + 1
        decision_nodes = (
            ast.If, ast.For, ast.While, ast.And, ast.Or,
            ast.ExceptHandler, ast.With, ast.IfExp, ast.BoolOp
        )
        
        count = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, decision_nodes):
                count += 1
        
        return count
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        
        def traverse(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            # Statements that increase nesting
            nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try)
            
            for child in ast.iter_child_nodes(node):
                new_depth = current_depth + (1 if isinstance(child, nesting_nodes) else 0)
                traverse(child, new_depth)
        
        traverse(tree)
        return max_depth
    
    def _extract_fallback_features(self, code: str) -> List[float]:
        """Regex-based fallback for non-Python languages."""
        features = {}
        
        lines = code.split('\n')
        total_lines = len(lines)
        
        # === TREE STRUCTURE (approximations) ===
        # Use braces/brackets for depth
        brace_depths = []
        current_depth = 0
        breadth_counts = {}
        
        for char in code:
            if char in '{[(':
                current_depth += 1
                brace_depths.append(current_depth)
                breadth_counts[current_depth] = breadth_counts.get(current_depth, 0) + 1
            elif char in '}])':
                current_depth = max(0, current_depth - 1)
        
        features['max_depth'] = max(brace_depths) if brace_depths else 0
        features['avg_depth'] = np.mean(brace_depths) if brace_depths else 0
        features['depth_std'] = np.std(brace_depths) if len(brace_depths) > 1 else 0
        features['max_breadth'] = max(breadth_counts.values()) if breadth_counts else 0
        features['tree_balance'] = (
            features['avg_depth'] / features['max_depth']
            if features['max_depth'] > 0 else 0
        )
        
        # === NODE STATISTICS ===
        # Approximate with keywords + identifiers
        keywords = re.findall(r'\b(if|else|for|while|switch|case|try|catch|finally|class|function|def|return|break|continue)\b', code)
        features['total_nodes'] = len(keywords) * 2  # Rough approximation
        features['unique_node_ratio'] = len(set(keywords)) / len(keywords) if keywords else 0
        features['node_entropy'] = 3.0  # Neutral default
        
        # === CONTROL FLOW ===
        features['if_count'] = len(re.findall(r'\bif\s*\(', code))
        features['loop_count'] = len(re.findall(r'\b(for|while)\s*\(', code))
        features['try_count'] = len(re.findall(r'\btry\s*[{\(]', code))
        features['cyclomatic_complexity'] = (
            1 + features['if_count'] + features['loop_count']
        )
        features['nesting_depth'] = features['max_depth']
        
        # === FUNCTION METRICS ===
        # Pattern: return_type function_name(params) {
        functions = re.findall(r'\b\w+\s+(\w+)\s*\([^)]*\)\s*{', code)
        features['num_functions'] = len(functions)
        
        # Approximate function lengths
        function_starts = [m.start() for m in re.finditer(r'\b\w+\s+\w+\s*\([^)]*\)\s*{', code)]
        if len(function_starts) > 0:
            func_lengths = []
            for i, start in enumerate(function_starts):
                end = function_starts[i + 1] if i + 1 < len(function_starts) else len(code)
                func_code = code[start:end]
                func_lengths.append(len(func_code.split('\n')))
            
            features['avg_function_length'] = np.mean(func_lengths)
            features['max_function_length'] = max(func_lengths)
        else:
            features['avg_function_length'] = 0
            features['max_function_length'] = 0
        
        # Function parameters
        param_patterns = re.findall(r'\(([^)]*)\)', code)
        param_counts = [len([p for p in params.split(',') if p.strip()]) 
                       for params in param_patterns if params.strip()]
        features['avg_function_params'] = np.mean(param_counts) if param_counts else 0
        features['max_function_params'] = max(param_counts) if param_counts else 0
        features['avg_function_complexity'] = features['cyclomatic_complexity'] / max(features['num_functions'], 1)
        
        # === CLASS METRICS ===
        classes = re.findall(r'\bclass\s+\w+', code)
        features['num_classes'] = len(classes)
        features['avg_methods_per_class'] = (
            features['num_functions'] / len(classes) if classes else 0
        )
        features['inheritance_depth'] = len(re.findall(r'extends|implements|:', code)) / max(len(classes), 1)
        
        # === DOCUMENTATION ===
        # Single-line comments
        single_comments = len(re.findall(r'//.*$', code, re.MULTILINE))
        # Multi-line comments
        multi_comments = len(re.findall(r'/\*.*?\*/', code, re.DOTALL))
        
        total_comments = single_comments + multi_comments
        features['docstring_ratio'] = (
            total_comments / max(features['num_functions'], 1)
        )
        features['comment_density'] = total_comments / total_lines if total_lines > 0 else 0
        features['avg_docstring_length'] = 50.0  # Default
        
        # === TYPE HINTS ===
        # Approximate with type annotations
        type_hints = len(re.findall(r':\s*\w+', code))
        features['type_hint_ratio'] = type_hints / max(features['num_functions'] * 2, 1)
        features['return_annotation_ratio'] = 0.3  # Default
        
        # === EXCEPTION HANDLING ===
        exceptions = re.findall(r'catch\s*\(\s*(\w+)', code)
        specific = sum(1 for exc in exceptions if exc not in ['Exception', 'Error', 'e'])
        features['exception_specificity'] = specific / len(exceptions) if exceptions else 0.5
        features['try_coverage'] = features['try_count'] / max(features['num_functions'], 1)
        
        # === NAMING PATTERNS ===
        identifiers = re.findall(r'\b[a-zA-Z_]\w+\b', code)
        if identifiers:
            snake = sum(1 for i in identifiers if '_' in i)
            camel = sum(1 for i in identifiers if any(c.isupper() for c in i))
            features['naming_consistency'] = max(snake, camel) / len(identifiers)
            features['avg_identifier_length'] = np.mean([len(i) for i in identifiers])
        else:
            features['naming_consistency'] = 0
            features['avg_identifier_length'] = 0
        
        # === CODE ORGANIZATION ===
        imports = re.findall(r'\b(import|include|using|require)\s+', code)
        features['import_count'] = len(imports)
        features['import_diversity'] = 0.7  # Default
        
        return [features[name] for name in self.feature_names]
    
    def _default_features(self) -> List[float]:
        """Return default features when extraction fails."""
        return [0.0] * len(self.feature_names)


if __name__ == "__main__":
    # Test the extractor
    extractor = EnhancedASTExtractor()
    
    test_code = """
def calculate_sum(numbers: list) -> int:
    \"\"\"Calculate the sum of a list of numbers.\"\"\"
    total = 0
    for num in numbers:
        if num > 0:
            total += num
    return total

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        try:
            self.data.append(item)
        except ValueError as e:
            print(f"Error: {e}")
"""
    
    features = extractor.extract(test_code, 'python')
    print("Extracted features:", len(features))
    for name, value in zip(extractor.feature_names, features):
        print(f"  {name}: {value:.4f}")
