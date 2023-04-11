

import pandas as pd
from base.base_cfg import BaseCfg
from sklearn.base import BaseEstimator, TransformerMixin

logger = BaseCfg.getLogger(__name__)


class RowFuncTransformer(BaseEstimator, TransformerMixin):
    """Drop rows with missing values.
    """

    def __init__(self,
                 func,
                 col: str,
                 inverse_func=None,
                 new_col: list[str] or str = None):
        self.func = func
        self.inverse_func = inverse_func
        self.col = col
        self.new_col = new_col

    def fit(self, X, y=None):
        # logger.debug(f'fit {self.col}')
        return self

    def transform(self, X):
        # logger.debug('transform {}'.format(target_col))
        if self.new_col is None:
            target_col = self.col
        else:
            target_col = self.new_col
        X1 = X.apply(
            lambda row: self.func(row, row[self.col]), axis=1)
        # logger.debug(X1)
        X.loc[:, target_col] = pd.DataFrame(list(X1))
        return X
        # return pd.DataFrame({self.col: X.values})
