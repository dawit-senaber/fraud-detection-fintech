# src/utils/transformers.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom log transformer with feature naming support"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return [f"log_feature_{i}" for i in range(self.n_features_in_)]
        return [f"log_{name}" for name in input_features]
