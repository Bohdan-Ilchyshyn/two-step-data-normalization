from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class FirstCustomScaler(BaseEstimator, ClassifierMixin):

    def __init__(self, name='first_custom_scaler'):
        self.name = name
        self._fitted: bool = False
        self._scale: Optional[float] = None

    def fit(self, X: pd.DataFrame):
        scale = X.abs().max()
        scale = scale.replace(0, 1)
        self._scale = scale
        self._fitted = True

    def transform(self, X: pd.DataFrame, skip_first_step: bool = False):
        self._check_is_fitted()
        X = X.copy()

        if not skip_first_step:
            X = self._norm_by_columns(X)

        X = self._norm_by_rows(X)
        return X

    def _check_is_fitted(self):
        if not self._fitted:
            raise TypeError(
                "This instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

    def _norm_by_columns(self, X):
        X = X / self._scale
        return X

    def _norm_by_rows(self, X):
        X = X.divide(
            np.sqrt(
                np.sum(
                    np.power(X, 2),
                    axis=1
                )
            ),
            axis='rows'
        )
        return X


class SecondCustomScaler(BaseEstimator, ClassifierMixin):

    def __init__(self, name='second_custom_scaler') -> None:
        self.name = name
        self._fitted: bool = False
        self._scale: Optional[float] = None

    def fit(self, X: pd.DataFrame) -> None:
        scale = X.abs().max()
        scale = scale.replace(0, 1)
        self._scale = scale
        self._fitted = True

    def transform(self, X: pd.DataFrame):
        self._check_is_fitted()
        X = X.copy()

        X = self._norm_by_columns(X)

        norms = self._calculate_norms(X)
        X = self._divide_by_norms(X, norms)
        X['norm'] = norms

        norms = self._calculate_norms(X)
        X = self._divide_by_norms(X, norms)

        return X

    def _check_is_fitted(self):
        if not self._fitted:
            raise TypeError(
                "This instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

    def _norm_by_columns(self, X):
        X = X / self._scale
        return X

    def _calculate_norms(self, X):
        return np.sqrt(
            np.sum(
                np.power(X, 2),
                axis=1
            )
        )

    def _divide_by_norms(self, X, norms):
        X = X.divide(norms, axis='rows')
        return X
