

from datetime import datetime
from enum import Enum
import pandas as pd
from base.base_cfg import BaseCfg
from base.timer import Timer
from helper.feature_select import FeatureSelector
from data.data_source import DataSource
from data.estimate_scale import EstimateScale
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from base.util import print_dateframe
from sklearn.metrics import explained_variance_score, accuracy_score, r2_score
from base.model_file_store import FileStore
from numpy import mean, std
from sklearn.feature_selection import SelectKBest, f_regression

logger = BaseCfg.getLogger(__name__)


class ModelClass(Enum):
    Regression = 'Regression'
    Classification = 'Classification'


class BasePredictor(BaseEstimator):
    """Predictor class

    Parameters
    ==========
    x_columns: list[str] collection of columns to be used as features
    x_required: list[str] collection of required columns. If not specified, x_columns is used.

    """

    def __init__(
        self,
        name: str = 'BasePredictor',
        data_source: DataSource = None,
        scale: EstimateScale = None,

        model=None,
        model_params: dict = None,
        x_columns: list[str] = None,
        x_required: list[str] = None,
        y_column: str = None,
        y_numeric_column: str = None,
        source_filter_func: (pd.Series) = None,
        source_date_span: int = None,
        source_suffix_list: list[str] = None,
        model_store=None,
        model_class: ModelClass = ModelClass.Regression,
        prefer_estimated_value: bool = True,
        logger=None
    ) -> None:
        self.name = name if name else self.__class__.__name__
        self.data_source = data_source
        self.scale = scale
        self.model = model
        self.model_params = model_params
        self.x_columns = x_columns
        self.x_required = x_required
        self.y_column = y_column
        self.y_numeric_column = y_numeric_column
        self.source_filter_func = source_filter_func
        self.source_date_span = source_date_span
        self.source_suffix_list = source_suffix_list
        self.model_store = model_store
        self.model_class = model_class
        self.prefer_estimated_value = prefer_estimated_value
        self._init_params()

    def _init_params(self):
        """Initialize the parameters"""
        self._name = self.name if self.name else self.__class__.__name__
        self._x_required = self.x_required if self.x_required else self.x_columns
        self._y_numeric_column = self.y_numeric_column if self.y_numeric_column else self.y_column
        if self.model_store is None:
            self.store = FileStore()
        else:
            self.store = self.model_store
        self.pred_accuracy_score = None
        self.df: pd.DataFrame = None
        self.df_prepared = None
        self.trained_seconds = 0
        self.predict_per_second = 0
        self.logger = logger or BaseCfg.getLogger(self._name)

    def set_params(self, **params):
        ret = super().set_params(**params)
        self._init_params()
        return ret

    def load_model(self):
        """Load the model"""
        data, accuracy, meta = self.store.load_model(self._name)
        self.pred_accuracy_score = accuracy
        self.model = data
        for k, v in meta.items():
            setattr(self, k, v)
        return self.model

    def save_model(self, model=None):
        """Save the model"""
        if model is None:
            model = self.model
        if model is None:
            raise ValueError('Model is not specified.')
        self.model_path = self.store.save_model(
            self._name,
            model,
            self.pred_accuracy_score,
            self.get_meta(),
        )
        return self.model_path

    def get_meta(self) -> dict:
        return {
            'name': self._name,
            'accuracy_score': self.pred_accuracy_score,
            'created_ts': self.created_ts,
            'trained_seconds': self.trained_seconds,
            'predict_per_second': self.predict_per_second,
            'model_params': self.model_params,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'x_columns': self.x_columns,
            'x_required': self._x_required,
        }

    def set_scale(self, scale: EstimateScale):
        """Set the scale"""
        self.scale = scale
        # need to reload the data
        self.df = None
        self.df_prepared = None

    def load_data(
        self,
        date_span: int = None,
        filter_func: (pd.Series) = None,
        suffix_list: list[str] = None,
    ) -> pd.DataFrame:
        """Load the data"""
        if self.data_source is None:
            raise ValueError('Data source is not specified.')
        if self.x_columns is None:
            col_list = None
        else:
            col_list = self.x_columns.copy()
            col_list.append(self.y_column)
        self.df = self.data_source.get_df(
            scale=self.scale,
            cols=col_list,
            date_span=date_span or self.source_date_span,
            filter_func=filter_func or self.source_filter_func,
            suffix_list=suffix_list or self.source_suffix_list,
        )
        self.logger.info(f'Data loaded. {self.source_suffix_list}')
        self.generate_numeric_columns()
        return self.df

    def prepare_data(self, X, params=None):
        """Prepare the data for training or prediction. 
        To be implemented by subclasses.
        """
        self.generate_numeric_columns()
        return X

    def prepare_model(self):
        """Prepare the model.
        To be implemented by subclasses.
        """
        self.created_ts = datetime.now()

    def generate_numeric_columns(self):
        """Return the number of columns"""
        self.x_numeric_columns_ = []
        for col in self.df.columns:
            try:
                if (col != self._y_numeric_column) and \
                    (self.df[col].dtype == 'float64' or
                     self.df[col].dtype == 'int64'):
                    self.x_numeric_columns_.append(col)
            except Exception as e:
                self.logger.error(
                    f'Error in generating numeric column:{col} error:{e}')
        if self.prefer_estimated_value:
            new_cols = self.x_numeric_columns_.copy()
            for col in self.x_numeric_columns_:
                if col.endswith('-e'):
                    orig_col = col[:-2] + '_n'
                    if orig_col in self.x_numeric_columns_:
                        new_cols.remove(orig_col)
            self.x_numeric_columns_ = new_cols
        return self.x_numeric_columns_

    # feature selection
    def select_features(self):
        feature_selector = FeatureSelector(
            self.df, y_col=self._y_numeric_column)
        x_columns_list = feature_selector.find_raw_features()

    def train(self):
        """Train the model"""
        timer = Timer(self._name, self.logger)
        # get the data
        if self.df_prepared is None:
            if self.df is None:
                self.load_data()
            # select features to use

            self.df_prepared = self.prepare_data(self.df)
        # split the data
        self.df_train, self.df_test = train_test_split(
            self.df_prepared, test_size=0.15, random_state=10)

        # X = self.df_prepared[self.x_columns].values
        # y = self.df_prepared[self.y_column].values
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=0)
        self.prepare_model()
        self.fit(
            self.df_train[self.x_numeric_columns_],
            self.df_train[self._y_numeric_column])
        self.pred_accuracy_score = self.test(
            X=None,
            X_test=self.df_test[self.x_numeric_columns_],
            y=self.df_test[self._y_numeric_column])
        self.trained_seconds = timer.stop(self.df_train.shape[0])
        # print(f'{self._name} accuracy: {self.pred_accuracy_score}')
        return self.pred_accuracy_score

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model"""
        self.model.fit(X, y)
        pass

    def cross_val(self, X=None, y=None):
        if X is None or y is None:
            X = self.df_test[self.x_numeric_columns_]
            y = self.df_test[self._y_numeric_column]
        # cross validation with k-fold
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(
            self.model,
            X,
            y,
            scoring='neg_mean_absolute_error',
            cv=cv,
            n_jobs=-1,
            error_score='raise'
        )
        self.cv_mean = mean(n_scores)
        self.cv_std = std(n_scores)
        self.logger.info(f'MAE: {self.cv_mean:.3f} ({self.cv_std:.3f})')

    def test(self, X, y, X_test=None):
        """Calculate the accuracy of the model."""
        if X_test is None:
            X_test = self.prepare_data(X)
        y_pred = self.predict(X_test)
        self.pred_accuracy_score = round(self.get_score(y, y_pred) * 10000)
        return self.pred_accuracy_score

    def get_score(self, y_true, y_pred):
        """Get the accuracy score."""
        if self.model_class is None or self.model_class == ModelClass.Regression:
            pred_accuracy_score = r2_score(y_true, y_pred)
        elif self.model_class == ModelClass.Classification:
            pred_accuracy_score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(
                f'Model class is not specified. {self.model_class}')
        return pred_accuracy_score

    def tune(self, date_spans: list[int] = None):
        """Tune the model. Two levels of tuning: 1. date_span 2. model_params"""
        if date_spans is None:
            # 6months, 1year, 2years, 4years, 8years
            date_spans = [180, 365, 730, 1461, 2922]
        best_date_span = 0
        best_accuracy_score = 0
        for date_span in date_spans:
            self.df_prepared = None
            self.load_data(date_span)
            # TODO use different model_params
            self.train()
            if self.pred_accuracy_score > best_accuracy_score:
                best_date_span = date_span
                best_accuracy_score = self.pred_accuracy_score
            # self.save_model()
        self.logger.info(
            f'{self._name} best date_span: {best_date_span} accuracy: {best_accuracy_score}', self.scale)
        return best_date_span, best_accuracy_score

    def tune_model(self, X, y, params, data_params=None):
        """Tune the model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=0)
        X_train = self.prepare_data(X_train, data_params)
        X_test = self.prepare_data(X_test, data_params)
        scores = [0] * len(params)
        for i in range(len(params)):
            self.model_param = params[i]
            self.train(None, y_train, X_train=X_train)
            scores[i] = self.test(None, y_test, X_test=X_test)
        max_accuracy_index = scores.index(max(scores))
        max_accuracy_params = params[max_accuracy_index]
        self.model_param = params
        return (scores, max_accuracy_index, max_accuracy_params)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the results"""
        return self.model.predict(X)

    def __str__(self) -> str:
        return self.__class__.__name__
