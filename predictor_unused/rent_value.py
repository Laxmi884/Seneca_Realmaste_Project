import copy
from math import isnan


import pandas as pd
from predictor.LGBMRegressor import LGBMRegressorPredictor
from predictor.writeback_mixin import WriteBackMixin
from data.data_source import DataSource

from base.base_cfg import BaseCfg
from base.util import print_dateframe
from data.estimate_scale import EstimateScale, PropertyType
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer

logger = BaseCfg.getLogger(__name__)


def needRent(row):
    return not (row['sp-n'] is None or isnan(row['sp-n']) or row['sp-n'] == 0)


class RentValueEstimator(LGBMRegressorPredictor, WriteBackMixin):
    """Value Estimator class"""

    def __init__(
        self,
        data_source: DataSource,
        scale: EstimateScale = None,
        model_store=None,
    ):
        super().__init__(
            name='RentValueEstimator',
            data_source=data_source,
            source_filter_func=needRent,
            source_date_span=365*18,
            source_suffix_list=['-n', '-c', '-b', '-e'],  # , '-b'],
            scale=scale,
            model_store=model_store,
            y_numeric_column='sp-n',
            y_column='sp',
            x_columns=[
                'onD',
                'cmty',
                'st', 'st_num', 'lat', 'lng',  'zip',
                'pstyl', 'ptp',
                'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
                'bths', 'rms', 'bsmt', 'schools', 'zone',
                'gr', 'tgr', 'gatp',
                'balcony', 'ac',
                'den_fr', 'ens_lndry', 'fce', 'lkr',
                'sqft', 'bltYr',
                'daddr',  'park_fac',
                'rltr',
                'pets', 'laundry', 'laundry_lev',
                'cac_inc', 'comel_inc', 'heat_inc', 'prkg_inc',
                'hydro_inc', 'water_inc', 'all_inc', 'pvt_ent',
                'insur_bldg', 'prkg_inc', 'tv',
            ],
        )
        # 'ptype2', 'rmSqft', 'rmBltYr',

        # 'offD', 'sldd','heat', 'feat', 'constr',
        # 'depth', 'flt','commuId',
        # 'tax', 'taxyr', 'mfee',
        # 'unt','ptype','saletp',
        # 'la', 'la2','comm',
        # 'lst',
        # 'lp', 'lpr', 'sp',

    def writeback(self):
        return super().writeback(
            new_col='rent_value-e')


class RentValue(LGBMRegressorPredictor):
    """Value Estimator class"""

    def __init__(
        self,
        data_source: DataSource,
        scale: EstimateScale = None,
        model_store=None,
    ):
        super().__init__(
            name='RentValue',
            data_source=data_source,
            source_filter_func=needRent,
            source_date_span=365*18,
            source_suffix_list=['-n', '-c', '-b', '-e'],  # , '-b'],
            scale=scale,
            model_store=model_store,
            y_numeric_column='sp-n',
            y_column='sp',
            x_columns=[
                'onD',
                'cmty',
                'st', 'st_num', 'lat', 'lng',  'zip',
                'pstyl', 'ptp',
                'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
                'bths', 'rms', 'bsmt', 'schools', 'zone',
                'gr', 'tgr', 'gatp',
                'balcony', 'ac',
                'den_fr', 'ens_lndry', 'fce', 'lkr',
                'sqft', 'bltYr',
                'daddr',  'park_fac',
                'rltr',
                'pets', 'laundry', 'laundry_lev',
                'cac_inc', 'comel_inc', 'heat_inc', 'prkg_inc',
                'hydro_inc', 'water_inc', 'all_inc', 'pvt_ent',
                'insur_bldg', 'prkg_inc', 'tv',
            ],
        )
