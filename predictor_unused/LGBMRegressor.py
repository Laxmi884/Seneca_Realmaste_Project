

import copy

from matplotlib.colorbar import ColorbarBase
from base.base_cfg import BaseCfg
from predictor.base_predictor import BasePredictor
import lightgbm as lgb

_logger = BaseCfg.getLogger(__name__)


class LGBMRegressorPredictor(BasePredictor):
    base_model_params = {
        'n_estimators': 300,
        'max_depth': -1,
        'num_leaves': 100,
    }

    def prepare_model(self, reset: bool = False):
        super().prepare_model()
        if reset or self.model is None:
            if self.model_params is None:
                self.model_params = copy.copy(
                    LGBMRegressorPredictor.base_model_params)
            self.model = lgb.LGBMRegressor(**self.model_params)
            self.logger.info('model_params: {}'.format(
                self.model_params))

    def prepare_data(self, X, params=None, logger=None):
        """Prepare data for training"""
        X = X.copy()
        rowsBefore = X.shape[0]
        colsBefore = X.shape[1]
        X.dropna(axis='columns', how='all', inplace=True)
        X = X.loc[:, (X != 0).any(axis=0)]
        colsAfter = X.shape[1]
        X.dropna(inplace=True)
        rowsAfter = len(X.index)
        self.logger.info(
            f'*{self.name}* Rows dropped: {rowsBefore-rowsAfter}/{rowsBefore}=>{rowsAfter}\
 Cols dropped: {colsBefore-colsAfter}/{colsBefore}=>{colsAfter}')
        self.generate_numeric_columns()
        return X
