
import copy
import pandas as pd
from base.base_cfg import BaseCfg
from sklearn.base import BaseEstimator, TransformerMixin

logger = BaseCfg.getLogger(__name__)


class BinaryTransformer(TransformerMixin, BaseEstimator):
    """Transforms string to 0 or 1

    Parameters
    ----------
    map: list of strings of two values
        the list of values mapping to 0 and 1

    Attributes
    ----------
    n_values_ : int
        The number of unique values of the data passed to :meth:`fit`.
    """

    def __init__(
        self,
        map: list,
        col: str,
    ):
        self.map = map
        self.col = col

    def get_feature_names_out(self):
        return [self.col+'-b']

    # def get_params(self, deep=True):
    #     params = {
    #         'col': self.col,
    #         'map': copy.deepcopy(self.map)
    #     }
    #     return params

    # def set_params(self, map: dict, col: str):
    #     self.map = map if map is not None else self.map
    #     self.col = col if col is not None else self.col
    #     if not isinstance(self.map, list) or len(self.map) != 2:
    #         raise ValueError(
    #             f'The map must be a list of two values, got {self.map}')
    #     return self

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
        if not isinstance(self.map, list) or len(self.map) != 2:
            raise ValueError(
                f'The map must be a list of two values, got {self.map}')
        # logger.debug(f'binary.fit {self.col}')
        self.n_values_ = len(X.value_counts())
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
        # logger.debug(f'binary.transform {self.col}')
        if isinstance(self.map, list):
            X = [1 if x == self.map[1] else 0 for x in X]
        else:
            raise ValueError(
                f'The map must be a list of two values, got {self.map}')
        return X
