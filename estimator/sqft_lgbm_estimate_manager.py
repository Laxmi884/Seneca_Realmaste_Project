from data.estimate_scale import EstimateScale
from estimator.lgbm_estimate_manager import LgbmEstimateManager
import pandas as pd
from math import isnan

from estimator.writeback_mixin import WritebackMixin

ESTIMATOR_NAME = 'Sqft'


def needSqft(row):
    return not (row['sqft-n'] is None or isnan(row['sqft-n']))


class SqftLgbmEstimateManager(LgbmEstimateManager, WritebackMixin):
    """Sqft estimate manager."""

    y_db_col = 'sqft_m1'
    y_column = 'sqft'
    x_columns = [
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
    ] # add more columns here
    
    #additional_x_cols = ['all_inc_N', 'all_inc_Y', 'cac_inc_N', 'cac_inc_Y', 'comel_inc_N', 'comel_inc_Y', 'daddr_N', 'daddr_Y', 'den_fr_N', 'den_fr_Y', 'heat_inc_N', 'heat_inc_Y', 'hydro_inc_N', 'hydro_inc_Y', 'insur_bldg_N', 'insur_bldg_Y', 'prkg_inc_N', 'prkg_inc_Y', 'pvt_ent_N', 'pvt_ent_Y', 'status_A', 'status_U', 'water_inc_N', 'water_inc_Y']

    def __init__(self, data_source):
        super().__init__(
            data_source,
            name=ESTIMATOR_NAME,
            model_params=None,
            min_output_value=150, #
            max_output_value=100000,
        )

    def my_load_data(self, scale: EstimateScale = None) -> pd.DataFrame:
        return self.load_data(
            scale=scale,
            filter_func=needSqft,
        )
