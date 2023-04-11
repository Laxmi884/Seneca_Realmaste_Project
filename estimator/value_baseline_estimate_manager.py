from base.const import MODEL_TYPE_REGRESSION, TRAINING_MIN_ROWS
from base.timer import Timer
from data.estimate_scale import EstimateScale
from estimator.rmbase_estimate_manager import RmBaseEstimateManager
import pandas as pd
from math import isnan
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score

###
# Calculate the value from the baseline statistics.
#   1. Calculate the value from all the features.
#   2. Use a model to calculate the value from the featured values.
###


DISCRETE_FEATURE_COLUMNS = [
    'gr',
    'tgr',
    'bdrms',
    'br_plus',
    'bthrms',
    'kch',
    ['sldd_Month', 'onD_Month'],  # in 2 year
]

CONTINUOUS_FEATURE_COLUMNS = [
    'flt',  # for detached only
    'lotSz',  # for detached only
    'lvl',  # for condos only
    'rmSz',
    'tax',
    'mfee',
    ['bltYr', 'rmBltYr', 'bltYr_m1'],
    ['sqft', 'rmSqft', 'sqft_m1'],
]


class ValueBaselineEstimateManager(RmBaseEstimateManager):
    """Value Baseline estimate manager."""

    model_name = 'baseline'
    date_span = 365 * 4
    suffix_list = ['-n', '-c', '-b', '-e']
    numeric_columns_only = True
    default_model_params = {
    }

    def __init__(
        self,
        data_source,
        name,
        model_params,
    ):
        super().__init__(
            data_source,
            name,
            model_class=MODEL_TYPE_REGRESSION,
        )
        self.model_params = model_params

    def prepare_model(self):
        """Prepare model."""
        model_params = self.model_params or self.default_model_params
        # TODO: implement baseline model
        self.model = None
        self.logger.info('model_params: {model_params}')
        return self.model

    def filter_data(
        self,
        X,
    ):
        """ Filter data for LGBMRegressor """
        return X

    def train_single_scale(self, scale: EstimateScale) -> tuple[EstimateScale, object, float, list[str], dict]:
        timer = Timer(str(scale), self.logger)
        timer.start()
        df = self.my_load_data(scale)
        if df is None or df.empty or df.shape[0] < TRAINING_MIN_ROWS:
            return (None, None, None, None, None, None)
        df_train, df_test = train_test_split(
            df, test_size=0.15, random_state=10)
        # TODO: do stats on df, do test on df_test
        model = self.prepare_model()
        x_cols, y_col, x_means = self.get_x_y_columns(df)
        model.fit(df_train[x_cols], df_train[y_col])
        accuracy = self.test_accuracy(
            model, df_test[x_cols], df_test[y_col])
        timer.stop(df_train.shape[0])
        self.logger.info('================================================')
        self.logger.info(
            f'{str(scale)} {str(self.model_name)} model trained accuracy:{accuracy/100.0}%')
        featureImportance = self.feature_importance(model)
        self.logger.info('------------------------------------------------')
        featureImportanceDic = {}
        for weight, feature in featureImportance:
            featureImportanceDic[feature] = int(weight)
            self.logger.info(f'{feature}: {weight}')
        return (scale, model, accuracy, x_cols, x_means, {'feature_importance': featureImportanceDic})

    def feature_importance(self, model) -> list:
        featureZip = list(zip(model.feature_importances_, model.feature_name_))
        featureZip.sort(key=lambda v: v[0], reverse=True)
        return featureZip

    def my_load_data(self, scale: EstimateScale = None) -> pd.DataFrame:
        """Subclass can override this method to load data.

        Args:
            scale (EstimateScale, optional): Defaults to None.

        Returns:
            pd.DataFrame: dataframe
        """
        return self.data_source.load_data(scale)
