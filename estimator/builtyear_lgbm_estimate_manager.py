from datetime import datetime
from base.timer import Timer
from base.util import dateToNum
from data.estimate_scale import EstimateScale
from estimator.lgbm_estimate_manager import LgbmEstimateManager
from math import isnan
import pandas as pd

from estimator.writeback_mixin import WritebackMixin


ESTIMATOR_NAME = 'BuiltYear'


def needBuiltYear(row):
    return not (row['bltYr-n'] is None or isnan(row['bltYr-n']))


class BuiltYearLgbmManager(LgbmEstimateManager, WritebackMixin):
    """Built year manager."""

    x_columns = [
        'lat', 'lng',
        'gatp', 'zip',
        'st_num-st-n',
        'bthrms', 'pstyl', 'ptp',
        'bsmt',  'heat', 'park_fac',
        'rms', 'bths',
        'st', 'st_num',
    ]
    y_column = 'bltYr'
    y_db_col = 'bltYr_m1'

    def __init__(self, data_source):
        super().__init__(
            data_source,
            name=ESTIMATOR_NAME,
            model_params=None,
            min_output_value=1800,
            max_output_value=datetime.now().year,
        )

    def my_load_data(self, scale: EstimateScale = None) -> pd.DataFrame:
        return self.load_data(
            scale=scale,
            filter_func=needBuiltYear,
        )
