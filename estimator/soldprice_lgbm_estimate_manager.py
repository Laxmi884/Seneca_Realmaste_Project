from data.estimate_scale import EstimateScale
from estimator.lgbm_estimate_manager import LgbmEstimateManager
import pandas as pd
from math import isnan

###
# TODO: Soldprice shall also consider the following factors:
#       - sold/asking price ratio for this type of property in the area
#       - sold/asking price ratio change over time
#       - value vs asking: a difference shall be calculated
#       - days on market: how it affects the sold/asking price ratio
###


ESTIMATOR_NAME = 'SoldPrice'


def needSold(row):
    return (not (row['sp-n'] is None or isnan(row['sp-n']))) and \
        (row['sp-n'] > 200000)


class SoldPriceLgbmEstimateManager(LgbmEstimateManager):
    """SoldPrice estimate manager."""

    y_target_col = 'sp-n-e'
    y_db_col = 'sp_m1'
    y_column = 'sp-n'
    x_columns = [
        'lp-n',
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

    def my_load_data(self, scale: EstimateScale = None) -> pd.DataFrame:
        scale = scale.copy(sale=True)
        return self.load_data(
            scale=scale,
            filter_func=needSold,
        )
