import copy
import datetime
from math import isnan


import pandas as pd
from predictor.LGBMRegressor import LGBMRegressorPredictor
from predictor.base_predictor import BasePredictor
from predictor.writeback_mixin import WriteBackMixin
from data.data_source import DataSource

from base.base_cfg import BaseCfg
from base.util import print_dateframe
from data.estimate_scale import EstimateScale, PropertyType
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer

import lightgbm as lgb

logger = BaseCfg.getLogger(__name__)


def needSqft(row):
    return not (row['sqft-n'] is None or isnan(row['sqft-n']))


class SqftEstimator(LGBMRegressorPredictor, WriteBackMixin):
    """Sqft Estimator class"""

    def __init__(
        self,
        data_source: DataSource,
        scale: EstimateScale = None,
        model_store=None,
    ):
        super().__init__(
            name='SqftEstimator',
            data_source=data_source,
            source_filter_func=needSqft,
            source_date_span=365*18,
            source_suffix_list=['-n', '-c'],  # , '-b'],
            scale=scale,
            model_store=model_store,
            y_numeric_column='sqft-n',
            y_column='sqft-n',
            x_columns=[
                'lat', 'lng',
                'bthrms', 'pstyl',
                'rms', 'bths',
                'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
                'zip',  'gatp',
                'depth', 'flt',
                'st_num-st-n',
                'ptp', 'pstyl', 'constr', 'feat',
                'bsmt',  'heat', 'park_fac',
                'st', 'st_num',
            ],
        )
        # 'sqft',
        # self.col_list = [
        #     'lat', 'lng', 'rmBltYr',
        #     'cs16', 'st_num', 'st', 'zip', 'sid',
        #     'gatp', 'flt', 'depth', 'gr', 'tgr', 'pstyl', 'bthrms',
        # ]
        # self.y_column = 'rmBltYr'
        # self.x_columns = [
        #     'lat', 'lng',
        #     'zip',  'gatp', 'bthrms',
        #     'st', 'st_num', 'st_num-st',
        # ]
        # self.categorical_feature = ['st', 'zip', 'gatp']

    def writeback(self):
        return super().writeback(
            new_col='sqft-e',
            orig_col='sqft-n')


class Sqft(LGBMRegressorPredictor):
    """Sqft Estimator class"""

    def __init__(
        self,
        data_source: DataSource,
        scale: EstimateScale = None,
        model_store=None,
    ):
        super().__init__(
            name='Sqft',
            data_source=data_source,
            source_filter_func=needSqft,
            source_date_span=365*18,
            source_suffix_list=['-n', '-c'],  # , '-b'],
            scale=scale,
            model_store=model_store,
            y_numeric_column='sqft-n',
            y_column='sqft-n',
            x_columns=[
                'lat', 'lng',
                'zip',  'gatp',
                'st', 'st_num',
                'bthrms', 'pstyl',
                'ptp', 'pstyl', 'constr', 'feat',
                'bsmt',  'heat', 'park_fac',
                'depth', 'flt', 'rms', 'bths',
                'sqft',
                'st_num-st-n',
            ],
        )
