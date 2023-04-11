from data.estimate_scale import EstimateScale
from estimator.lgbm_estimate_manager import LgbmEstimateManager
import pandas as pd
from math import isnan
from estimator.writeback_mixin import WritebackMixin

###
# TODO: single stage of estimator does not work well on the price change based on sold date.
#       Need to add more stages:
#         1. apply seasonal and monthly change/delta to the estimated value
#         2. apply feature delta to the estimated value (optional)
#         3. use another estimator to estimate the value based on the above adjustment.
###


ESTIMATOR_NAME = 'Value'


def needSold(row):
    return (not (row['sp-n'] is None or isnan(row['sp-n']))) and \
        (row['sp-n'] > 200000)


class ValueLgbmEstimateManager(LgbmEstimateManager, WritebackMixin):
    """Value estimate manager."""

    y_target_col = 'value-n-e'
    y_db_col = 'value_m1'
    y_column = 'sp-n'
    x_columns = [
        'onD',
        'cmty',
        'st', 'st_num', 'lat', 'lng',
        'pstyl', 'ptp',
        'tax', 'taxyr',
        'bdrms', 'tbdrms', 'br_plus', 'bthrms', 'kch', 'kch_plus',
        'bths', 'rms', 'bsmt', 'schools', 'zone',
        'gr', 'tgr', 'gatp',
        'depth', 'flt',
        'heat', 'feat', 'constr', 'balcony', 'ac',
        'den_fr', 'ens_lndry', 'fce',
        'sqft', 'bltYr',
        'park_fac',
    ]
    roundBy = 1000

    def __init__(self, data_source):
        super().__init__(
            data_source,
            name=ESTIMATOR_NAME,
            model_params=None,
            min_output_value=20000,
            max_output_value=10000000,
        )

    def load_scales(self, sale: bool = None) -> None:
        return super().load_scales(sale=True)

    def my_load_data(self, scale: EstimateScale) -> pd.DataFrame:
        scale = scale.copy(sale=True)
        return self.load_data(
            scale=scale,
            filter_func=needSold,
        )
