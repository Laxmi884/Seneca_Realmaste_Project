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


def needBuiltYear(row):
    return not (row['bltYr-n'] is None or isnan(row['bltYr-n']))


class BuiltYearEstimator(LGBMRegressorPredictor, WriteBackMixin):
    """Built Year Estimator class"""

    def __init__(
        self,
        data_source: DataSource,
        scale: EstimateScale = None,
        model_store=None,
    ):
        super().__init__(
            name='BuiltYearEstimator',
            data_source=data_source,
            source_filter_func=needBuiltYear,
            source_date_span=365*18,
            source_suffix_list=['-n', '-c', '-b'],
            scale=scale,
            model_store=model_store,
            y_numeric_column='bltYr-n',
            y_column='bltYr',
            x_columns=[
                'lat', 'lng',
                'gatp', 'zip',
                'st_num-st-n',
                'bthrms', 'pstyl', 'ptp',
                'bsmt',  'heat', 'park_fac',
                'rms', 'bths',
                'st', 'st_num',
            ],
        )
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
            new_col='bltYr-e',
            orig_col='bltYr-n')


class BuiltYear(LGBMRegressorPredictor):
    """Built Year Estimator class"""

    def __init__(
        self,
        data_source: DataSource,
        scale: EstimateScale = None,
        model_store=None,
    ):
        super().__init__(
            name='BuiltYear',
            data_source=data_source,
            source_filter_func=needBuiltYear,
            source_date_span=365*18,
            source_suffix_list=['-n', '-c', '-b'],
            scale=scale,
            model_store=model_store,
            y_numeric_column='bltYr-n',
            y_column='bltYr',
            x_columns=[
                'lat', 'lng',
                'zip',  'gatp',
                'st', 'st_num',
                'bthrms', 'pstyl',
                'ptp', 'pstyl', 'constr', 'feat',
                'bsmt',  'heat', 'park_fac',
                'depth', 'flt', 'rms', 'bths',
                'st_num-st-n',
            ],
        )
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
