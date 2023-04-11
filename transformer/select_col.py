import pandas as pd
from base.base_cfg import BaseCfg
from math import isnan
from sklearn.base import BaseEstimator, TransformerMixin

logger = BaseCfg.getLogger(__name__)


class SelectColumnTransformer(TransformerMixin, BaseEstimator):
    """Transforms string to 0 or 1

    Parameters
    ----------
    col: str
        the target column name
    columns: list[str] of strings of two or more
        The columns to be selected.
    type: default int. 
        The type of filter.

    Attributes
    ----------
    """

    def __init__(
        self,
        new_col: str,
        columns: list[str],
        func: (any),
        as_na_value=None
    ):
        self.new_col = new_col
        self.columns = columns
        self.func = func
        self.as_na_value = as_na_value

    def get_feature_names_out(self):
        return [self.new_col]

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        # logger.debug(f'fit {self.columns}')
        return self

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : {array-like, sparse matrix}, shape (n_samples, n_features_)
            The transformed data.
        """
        # logger.debug(f'transform {self.columns}')
        self.count_found = dict(zip(self.columns, [0]*len(self.columns)))
        self.count_found['not_found'] = 0

        def select_col(row):
            for col in self.columns:
                if col not in row:
                    continue
                v = self.func(row[col])
                if v is not None and not isnan(v):
                    self.count_found[col] += 1
                    #logger.debug(f'{col}:{row[col]}=> {v}')
                    return v
            self.count_found['not_found'] += 1
            return self.as_na_value
        X.loc[:, self.new_col] = X.apply(select_col, axis=1)
        # logger.debug(f'found {self.count_found}')
        return X
