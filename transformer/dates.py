
import datetime
from math import isnan
import re
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from base.util import dateFromNum, dateFromNumOrNow
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.const_label_map import getLevel

logger = BaseCfg.getLogger(__name__)
DATE_20000101 = datetime.datetime(2000, 1, 1, 0, 0)


class DatesTransformer(BaseEstimator, TransformerMixin):
    """Dates transformer.
    Transform dates to numeric values.

    Parameters
    ----------

    """
    maxLevel = 3

    def __init__(
        self,
        cols
    ):
        self.cols = cols

    def get_feature_names_out(self, cols: list[str] = None):

        if hasattr(self, '_target_cols') and self._target_cols is not None:
            return self._target_cols

        self._target_cols = [
            'onD-date',
            'onD-year-n',
            'onD-season-n',
            'onD-month-n',  # based on Janurary of previous year, could be negative
            'onD-week-n',
            # 'onD-dayOfWeek-n',
            # 'onD-dayOfMonth-n',
            # 'onD-dayOfYear-n',
        ]
        if cols is None:
            cols = self.cols
        if 'offD' in cols:
            self.offD = True
            self._target_cols.append('offD-date')
            self._target_cols.append('offD-year-n')
            self._target_cols.append('offD-season-n')
            self._target_cols.append('offD-month-n')
        else:
            self.offD = False
        if 'sldd' in cols:
            self.sldd = True
            self._target_cols.append('sldd-date')
            self._target_cols.append('sldd-dom-n')
        else:
            self.sldd = False
        return self._target_cols

    def set_params(self, **params):
        ret = super().set_params(**params)
        self._target_cols = None
        return ret

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
        # logger.debug(f'fit dates')

        self.get_feature_names_out()
        if 'sldd' in X.columns:
            self.sldd = True
        if 'offD' in X.columns:
            self.offD = True
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
        # logger.debug(f'transform dates')
        timer = Timer(self.__class__.__name__, logger)
        toAddCols = [col for col in self.get_feature_names_out()
                     if col not in X.columns]
        # X = pd.concat(
        #     [X, pd.DataFrame(columns=toAddCols, index=X.index)], axis=1)
        X[toAddCols] = 0
        hasOffD = 'offD' in X.columns
        hasSldd = 'sldd' in X.columns
        previous2Year = datetime.datetime.now().year - 2

        X['onD-date'] = X['onD'].apply(dateFromNum)
        X['onD-year-n'] = X['onD-date'].apply(lambda x: x.year)
        X['onD-season-n'] = X['onD-date'].apply(lambda x: x.month // 3)
        X['onD-month-n'] = X['onD-date'].apply(
            lambda x: x.month + (x.year - previous2Year) * 12)
        X['onD-week-n'] = X['onD-date'].apply(lambda x: x.isocalendar().week)
        # X['onD-2k-n'] = X['onD-date'].apply(lambda x: abs((x - DATE_20000101).days))
        # X['onD-dayOfWeek-n'] = X['onD-date'].apply(lambda x: x.weekday())
        # X['onD-dayOfMonth-n'] = X['onD-date'].apply(lambda x: x.day)
        # X['onD-dayOfYear-n'] = X['onD-date'].apply(lambda x: x.timetuple().tm_yday)
        if hasOffD:
            X['offD-date'] = X['offD'].apply(dateFromNumOrNow)
            X['offD-year-n'] = X['offD-date'].apply(lambda x: x.year)
            X['offD-season-n'] = X['offD-date'].apply(lambda x: x.month // 3)
            X['offD-month-n'] = X['offD-date'].apply(
                lambda x: x.month + (x.year - previous2Year) * 12)
        if hasSldd:
            X['sldd-date'] = X['sldd'].apply(dateFromNum)
            X['sldd-dom-n'] = (X['sldd-date'] - X['onD-date']
                               ).apply(lambda x: x.days)

        totalCount = X.shape[0]
        onDNanCount = X['onD'].isna().sum()
        if hasOffD:
            offDNanCount = X['offD'].isna().sum()
        else:
            offDNanCount = 0
        logger.info(
            f'onD|offD is None: {onDNanCount}|{offDNanCount}/{totalCount}')
        timer.stop(totalCount)
        return X
