"""TF-IDF feature extraction for code text."""

import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config as cfg


class CodeTFIDFExtractor:
    """TF-IDF features from raw code strings."""

    def __init__(self, max_features: int = None):
        self.max_features = max_features or cfg.TFIDF_DIM
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[a-zA-Z_]\w*|\d+|[^\s\w]",
            max_features=self.max_features,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, code_texts):
        """Fit on training code only."""
        self.vectorizer.fit(code_texts)
        self._fitted = True
        print(f"✓ TF-IDF fitted: {len(self.vectorizer.vocabulary_)} features")

    def transform(self, code_texts):
        """Transform code texts → sparse CSR matrix [N, max_features]."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self.vectorizer.transform(code_texts)

    def fit_transform(self, code_texts):
        """Fit and transform in one step. Returns sparse CSR matrix."""
        self.fit(code_texts)
        return self.transform(code_texts)

    def save(self, path: str = None):
        """Save fitted vectorizer to disk."""
        path = path or os.path.join(cfg.CACHE_DIR,
                                     f"tfidf_vectorizer_task{cfg.ACTIVE_TASK}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"✓ TF-IDF vectorizer saved: {path}")

    @classmethod
    def load(cls, path: str = None):
        """Load a previously fitted extractor."""
        path = path or os.path.join(cfg.CACHE_DIR,
                                     f"tfidf_vectorizer_task{cfg.ACTIVE_TASK}.pkl")
        ext = cls.__new__(cls)
        ext.vectorizer = joblib.load(path)
        ext.max_features = ext.vectorizer.max_features
        ext._fitted = True
        print(f"✓ TF-IDF vectorizer loaded: {path}")
        return ext
