

import pandas as pd
from base.base_cfg import BaseCfg
import numpy as np
from base.const import Mode
from sklearn.base import BaseEstimator, TransformerMixin


logger = BaseCfg.getLogger(__name__)


class DropRowTransformer(BaseEstimator, TransformerMixin):
    """Drop rows with missing values.
    Parameters
    ----------
    drop_cols : drop rows with missing values in these columns
    drop_func : drop rows when this function returns True. The function will be passed a row.
    as_na_value: str
        The value will be treated as missing value.
    columns: list
    inplace: bool
        If True, do operation inplace.
    """

    def __init__(
        self,
        drop_cols: list[str] = None,
        drop_func: (any) = None,
        as_na_value=None,
        columns=None,
        inplace=True
    ):
        self.drop_cols = drop_cols
        self.drop_func = drop_func
        self.as_na_value = as_na_value
        self.columns = columns
        self.inplace = inplace

    def get_feature_names_out(self):
        return self.columns

    def fit(self, X, y=None):
        # logger.debug('drop_na.fit')
        if self.drop_cols is None and self.drop_func is None:
            raise ValueError('Either drop_cols or drop_func must be specified')
        return self

    def transform(self, X):
        # logger.debug('drop_na.transform {}'.format(self.drop_cols))
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns)
            logger.info('converted np.ndarray to pd.DataFrame')
        # original_rows = X.shape[0]
        # logger.debug(f'{self.drop_cols} before {original_rows} rows')
        retX = None
        if self.drop_cols is not None:
            if self.as_na_value is None:
                retX = X.dropna(subset=self.drop_cols)
            else:
                retX = X[X[self.drop_cols] != self.as_na_value]
        elif self.drop_func is not None:
            if self.inplace:
                X.drop(
                    X.index[X.apply(lambda row: self.drop_func(row), axis=1)], inplace=True)
                retX = X
            else:
                retX = X.drop(
                    X.index[X.apply(lambda row: self.drop_func(row), axis=1)], inplace=False)
            # if self.inplace:
            #     retX = X
        else:
            raise ValueError('Either drop_cols or drop_func must be specified')
        # after_drop_rows = retX.shape[0]
        # logger.debug(
        #    f'{self.drop_cols} after {after_drop_rows}/{original_rows} dropped rows{original_rows - after_drop_rows}')
        return retX
