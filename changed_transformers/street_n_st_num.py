
from math import isnan
import re
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from numpy import NAN
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.const_label_map import getLevel

logger = BaseCfg.getLogger(__name__)


class StNumStTransformer(BaseEstimator, TransformerMixin):
    """bths transformer.

    Parameters
    ----------

    """
    maxLevel = 3

    def __init__(self):
        self._target_col = 'st_num-st-n'

    def get_feature_names_out(self):
        return [self._target_col]

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
        # logger.debug(f'fit st_num-st')
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
        # logger.debug(f'transform st_num-st')
        if 'st-c' not in X.columns or 'st_num-n' not in X.columns:
            logger.warning(f'st-c or st_num-n not in X.columns')
            return X
        timer = Timer('st_num-st', logger)
        nanCount = 0
        totalCount = 0
        # X[self._target_col] = NAN
        # for i, row in X.iterrows():
        #     totalCount += 1
        #     st = row['st-c']
        #     st_num = row['st_num-n']
        #     if (st is None) or (st_num is None) or isnan(st) or isnan(st_num):
        #         nanCount += 1
        #     X.loc[i, 'st_num-st-n'] = (st + 1) * 100000 + st_num
        X[self._target_col] = X['st-c'].astype(
            int, errors='ignore') * 100000 + X['st_num-n'].astype(int, errors='ignore')
        logger.info(
            f'{X[self._target_col].isna().sum()}/{X.shape[0]} nan values in st_num-st')
        timer.stop(totalCount)
        return X
