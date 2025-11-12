"""Custom transformers used across the KNN training and inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class SecondaryChoiceImputer(BaseEstimator, TransformerMixin):
    """Imputes the `secondary_choice` column using a classifier.

    The class trains a RandomForestClassifier (allowed by the project rules) to
    predict missing values based on the numeric attributes and the declared
    `primary_choice`. Numeric columns are temporarily filled with their median
    values only for the purpose of training/inference; the original frame keeps
    its NaNs so that downstream imputers can still model the numeric gaps.
    """

    numeric_features: Iterable[str]
    primary_feature: str = "primary_choice"
    target_feature: str = "secondary_choice"
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = 12

    def __post_init__(self) -> None:
        self._classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced_subsample",
        )
        self._primary_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._numeric_medians: Optional[pd.Series] = None
        self._should_skip = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        frame = X.copy()
        mask = frame[self.target_feature].notna()
        if mask.sum() == 0:
            self._should_skip = True
            return self

        feature_cols = list(self.numeric_features) + [self.primary_feature]
        working = frame.loc[mask, feature_cols + [self.target_feature]].copy()
        self._numeric_medians = working[self.numeric_features].median()

        X_cat = self._primary_encoder.fit_transform(working[[self.primary_feature]])
        X_num = working[self.numeric_features].fillna(self._numeric_medians).to_numpy()
        X_model = np.hstack([X_num, X_cat])
        y_model = working[self.target_feature].to_numpy()

        self._classifier.fit(X_model, y_model)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        if self._should_skip or frame[self.target_feature].isna().sum() == 0:
            frame[self.target_feature] = frame[self.target_feature].fillna("Unknown")
            return frame

        mask = frame[self.target_feature].isna()
        if mask.sum() == 0:
            return frame

        feature_cols = list(self.numeric_features) + [self.primary_feature]
        to_predict = frame.loc[mask, feature_cols].copy()
        X_cat = self._primary_encoder.transform(to_predict[[self.primary_feature]])
        medians = self._numeric_medians if self._numeric_medians is not None else to_predict[self.numeric_features].median()
        X_num = to_predict[self.numeric_features].fillna(medians).to_numpy()
        X_model = np.hstack([X_num, X_cat])

        frame.loc[mask, self.target_feature] = self._classifier.predict(X_model)
        return frame
