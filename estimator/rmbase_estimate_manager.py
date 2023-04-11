from datetime import datetime
from enum import Enum
from math import isnan
from typing import Union
from base.base_cfg import BaseCfg
from base.const import MODEL_TYPE_CLASSIFICATION, MODEL_TYPE_REGRESSION
from base.model_store import ModelStore
from base.util import expendList, getRoundFunction, logDataframeChange
from data.data_source import DataSource
import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, accuracy_score, r2_score

from data.estimate_scale import EstimateScale


class RmBaseEstimateManager:
    """RM base estimate manager.
    Each estimate manager is responsible for one type of estimate,
    and shall extend this class.

    attributes:
        x_columns: list[str]
        y_column: str
        date_span: int
        filter_func: pd.Series -> bool
        suffix_list: list[str]
        scales: {repr(scale): EstimateScale}
            model, meta, accuracy are stored in each scale.
        scale: EstimateScale
            when there is only one scale, this is the scale.
            either scale or scales is set.

    generated attributes:
        df: pd.DataFrame


    Parameters
    ==========
    DataSource: data.data_source.DataSource
    estimate_both: bool = False. True for both sale and rent.
    """

    def __init__(
        self,
        data_source: DataSource,
        name: str = None,
        model_class: str = MODEL_TYPE_REGRESSION,
        estimate_both: bool = False,
        min_output_value: float = 0,
        max_output_value: float = 1000000000,
    ) -> None:
        self.data_source = data_source
        #self.flag_to_include_else = data_source.flag_to_include_else # additional columns f
        self.name = name
        self.model_class = model_class
        self.logger = BaseCfg.getLogger(name or self.__class__.__name__)
        self.estimate_both = estimate_both
        self.min_output_value = min_output_value
        self.max_output_value = max_output_value
        pass

    def load_scales(self, sale: bool = None) -> None:
        """Load the scales or scale suitable for this estimator.
        This step must be done before train/save/load.
        To support other filters, override this method.
        """
        self.scales = {}
        for scale in self.data_source.getLeafScales(sale=sale):
            self.scales[repr(scale)] = scale

    def __model_key__(self) -> str:
        return f'{self.name}:{self.model_name}'

    def train(self) -> None:
        """Train the estimator.
        Train the estimator(s) for the specified scale or all scales.
        """
        if hasattr(self, 'scale'):
            scale, model, accuracy, x_cols, x_means, meta = self.train_single_scale(
                scale)
            if scale is None:
                self.logger.warning('No scale to train')
                return
            model_dict = {
                'model': model,
                'accuracy': accuracy,
                'x_cols': x_cols,
                'x_means': x_means,
                'feature_importance': meta['feature_importance'],
            }
            scale.meta[self.__model_key__()] = model_dict
        elif hasattr(self, 'scales'):
            for scale in self.scales.values():
                scale, model, accuracy, x_cols, x_means, meta = self.train_single_scale(
                    scale)
                if scale is None:
                    continue
                model_dict = {
                    'model': model,
                    'accuracy': accuracy,
                    'x_cols': x_cols,
                    'x_means': x_means,
                    'feature_importance': meta['feature_importance'],
                }
                scale.meta[self.__model_key__()] = model_dict
        else:
            self.logger.warning(
                'No scale or scales defined. Load scales first.')
            self.load_scales()
            self.train()

    def train_single_scale(self, scale: EstimateScale) -> tuple[EstimateScale, object, float, list[str], pd.Series, dict]:
        """Train the estimator for a single scale.
        This method is called by train method.
        """
        raise NotImplementedError

    def get_output_column(self) -> str:
        return getattr(self, 'y_target_col', (self.y_column + '-e'))

    def get_writeback_db_column(self) -> str:
        return getattr(self, 'y_db_col', None)

    def estimate(self, df_grouped: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Estimate the data source.
        Either train or load must be called before this.
        """
        if hasattr(self, 'scale'):
            return self.estimate_single_scale(
                df_grouped=df_grouped, scale=scale)
        elif hasattr(self, 'scales'):
            df_y_list = []
            y_cols_list = []
            y_db_cols_list = []
            for scale in self.scales.values():
                df_y, y_cols, y_db_cols = self.estimate_single_scale(
                    df_grouped=df_grouped, scale=scale)
                if df_y is not None:
                    df_y_list.append(df_y)
                    for i in range(len(y_cols)):
                        col = y_cols[i]
                        if col not in y_cols_list:
                            y_cols_list.append(col)
                            # y_db_cols_list must match y_cols_list
                            y_db_cols_list.append(y_db_cols[i])
            if len(df_y_list) > 0:
                df_y = pd.concat(df_y_list)
                return df_y, y_cols_list, y_db_cols_list
            else:
                return None, None, None
        else:
            self.logger.error(
                'No scale or scales defined.')
            raise Exception('No scale or scales defined.')

    def estimate_single_scale(
        self,
        df_grouped: pd.DataFrame,
        scale: EstimateScale
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Estimate the data source for a single scale.
        This method is called by estimate method.
        """
        # load data:
        # date_span=-1 means do not filter by date
        # sale is set to both to load all data
        if self.estimate_both:
            filterScale = scale.copy(sale='Both')
        else:
            filterScale = scale
        df = self.load_data(df_grouped=df_grouped,
                            scale=filterScale, date_span=-1)
        if df is None or (df.shape[0] == 0):
            # self.logger.warning(
            #     f'No data found for scale {scale}.')
            return None, None, None
        key = self.__model_key__()
        if key not in scale.meta:
            self.logger.warning(
                f'No model found for scale {scale}.')
            return None, None, None
        model_dict = scale.meta[key]
        model = model_dict['model']
        x_cols = model_dict['x_cols']
        x_means = model_dict['x_means']
        if x_means is None:
            raise Exception('x_means not found.')
        # fill missing values with mean
        for col in list(set(x_cols) - set(df.columns)):
            df[col] = x_means[col]
        self.logger.debug(df.head())
        y = model.predict(df[x_cols])
        if self.model_class == MODEL_TYPE_REGRESSION:
            y = self.round_result(y)
        self.logger.info(
            f'Estimation result for {self.name} {str(scale)} [{self.min_output_value_},{self.max_output_value_}]: {y}')
        y_target_col = self.get_output_column()
        df[y_target_col] = y
        df[y_target_col+'-acu'] = model_dict['accuracy']
        y_db_col = self.get_writeback_db_column()
        if y_db_col is not None:
            y_db_cols = [y_db_col, y_db_col+'_acu']
        else:
            y_db_cols = [None, None]
        return (df.loc[:, [y_target_col, y_target_col+'-acu']], [y_target_col, y_target_col+'-acu'], y_db_cols)

    def save(self, store: ModelStore) -> None:
        """Save the estimator(s).
        Shall save all the estimators. train or load must be called before this.
        """
        if hasattr(self, 'scale'):
            self.save_one_model(
                store, self.scale)
        elif hasattr(self, 'scales'):
            for scale in self.scales.values():
                self.save_one_model(store, scale)
        else:
            raise Exception('No scale or scales is set.')

    def save_one_model(self, store: ModelStore, scale: EstimateScale) -> None:
        """Save one estimator."""
        model_key = self.__model_key__()
        if model_key not in scale.meta:
            return None
        model_dict = scale.meta[model_key]
        # estimator name, model name, scale, date
        filename = ':'.join([self.name, self.model_name, repr(scale)])
        meta = {
            'accuracy': model_dict['accuracy'],
            'x_cols': model_dict['x_cols'],
            'x_means': model_dict['x_means'],
            'feature_importance': model_dict['feature_importance'],
            'ts': datetime.now(),
        }
        self.logger.info(f'Saving model: {filename} {meta}')
        store.save_model(
            filename, model_dict['model'], model_dict['accuracy'], meta)

    def load(self, store: ModelStore):
        """Load the estimator(s)."""
        if hasattr(self, 'scale'):
            scale, model, accuracy, meta = self.load_one_model(
                store, self.scale)
            meta['model'] = model
            meta['accuracy'] = accuracy
            self.scale.meta[self.__model_key__()] = meta
        elif hasattr(self, 'scales'):
            for scale in self.scales.values():
                scale, model, accuracy, meta = self.load_one_model(
                    store, scale)
                meta['model'] = model
                meta['accuracy'] = accuracy
                scale.meta[self.__model_key__()] = meta
        else:
            raise Exception('No scale or scales is set.')

    def load_one_model(self, store: ModelStore, scale: EstimateScale) -> tuple[EstimateScale, any, float, dict]:
        """Load one estimator."""
        filename = ':'.join([self.name, self.model_name, repr(scale)])
        try:
            model, accuracy, meta = store.load_model(filename)
        except FileNotFoundError as e:
            self.logger.error('Can not find model: %s', filename)
            raise e
        except Exception as e:
            self.logger.error('Failed to load model: %s', filename)
            raise e
        return (scale, model, accuracy, meta)

    # ---- Feature selection ----
    def select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features."""
        raise NotImplementedError

    def tune(self) -> None:
        """Tune the estimator."""
        raise NotImplementedError

    def test(self) -> None:
        """Test the estimator."""
        raise NotImplementedError

    # ---- Supporting functions ----
    def load_data(
        self,
        scale: EstimateScale,
        x_columns: list[str] = None,
        y_column: str = None,
        date_span: int = None,
        filter_func: (pd.Series) = None,
        suffix_list: list[str] = None,
        numeric_columns_only: bool = None,
        prefer_estimated: bool = False,
        df_grouped: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Load the data and set it to self.df.
        Use the default values if not specified:
        self.x_columns, self.y_column, self.date_span,
        self.filter_func, self.suffix_list
        """
        if self.data_source is None:
            raise ValueError('Data source is not specified.')
            
            
        col_list = []
        additional_columns = []
        
        if x_columns is not None:
            col_list.extend(x_columns)
        elif hasattr(self, 'x_columns'):
            # self.logger.debug(f'x_columns: {self.x_columns}')
            col_list.extend(self.x_columns)
            
        if len(col_list) == 0:
            # when no x_columns are specified, load all columns
            col_list = None
        elif y_column is not None:
            col_list.append(y_column)
        elif hasattr(self, 'y_column'):
            col_list.append(self.y_column)
            
        
        if hasattr(self, "additional_x_cols"): # additional str columns
            additional_columns.extend(self.additional_x_cols)
        
        #print(col_list)
        #exit()      
        # add more cols here

            
        numeric_columns_only = numeric_columns_only or getattr(
            self, 'numeric_columns_only', False)
        # do not save df to self.df, as subclases may need to load different data sets multiple times
        df = self.data_source.get_df(
            scale=scale,
            cols=col_list,
            date_span=date_span or self.date_span,
            filter_func=filter_func,  # self.filter_func: bound method does not work
            suffix_list=suffix_list or self.suffix_list,
            numeric_columns_only=numeric_columns_only,
            prefer_estimated=prefer_estimated,
            df_grouped=df_grouped,
            ad_cols = self.data_source.flag_to_include_else
        )
        # self.logger.info(f'Data loaded. {suffix_list or self.suffix_list}')
        return df

    def filter_data(
        self,
        X,
    ):
        """ Filter data. To be implemented by subclasses."""
        return X

    def get_x_y_columns(
        self,
        df: pd.DataFrame,
        y_column: str = None,
    ) -> tuple[list[str], str, pd.Series]:
        """Get numeric columns of x and y from df."""
        df = self.filter_data(df)
        #print("!")
        #print(df.info())
        #print("!")
        ##exit()
        #print()
        y_column = y_column or self.y_column
        y_numeric_column = None
        x_numeric_columns = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                x_numeric_columns.append(col)
                if col == y_column:
                    y_numeric_column = col
                elif col.startswith(y_column) and (col[-2:] in ['-b', '-n', '-c']) and ('-bl-' not in col) and ('-blm-' not in col):
                    y_numeric_column = col
        if y_numeric_column is None:
            raise ValueError(f'Column {y_column} is not numeric.')
        x_numeric_columns.remove(y_numeric_column)
        # self.logger.info(
        #     f'*{self.name}* X: {x_numeric_columns} y: {y_numeric_column}')
        x_means = df[x_numeric_columns].mean().to_dict()
        return x_numeric_columns, y_numeric_column, x_means

    def test_accuracy(self, model, X_test, y_test) -> float:
        """Calculate the accuracy of the model."""
        y_pred = self.round_result(model.predict(X_test))
        self.pred_accuracy_score = round(
            self.get_score(y_test, y_pred) * 10000)
        return self.pred_accuracy_score

    def round_result(self, y_pred: Union[pd.Series, pd.DataFrame, np.ndarray], col: str = None):
        fnRound = getRoundFunction(
            getattr(self, 'roundBy', 1),
            min=self.min_output_value_,
            max=self.max_output_value_,
        )
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.map(fnRound)
        elif isinstance(y_pred, pd.DataFrame):
            y_pred[col] = y_pred[col].map(fnRound)
        elif isinstance(y_pred, np.ndarray):
            fnRound = np.vectorize(fnRound)
            y_pred = fnRound(y_pred)
        else:
            raise ValueError(f'Unknown type: {type(y_pred)}')
        return y_pred

    def fit_output_min_max(self, y: pd.Series):
        """ use standard distribution to calculate the soft min and max output values.
        """
        stats = y.agg(['min', 'max', 'mean', 'std']).to_dict()
        min_output_value = min(
            stats['min'], stats['mean'] - 3 * stats['std'])
        self.min_output_value_ = max(min_output_value, self.min_output_value)
        max_output_value = max(
            stats['max'], stats['mean'] + 3 * stats['std'])
        self.max_output_value_ = min(max_output_value, self.max_output_value)

    def filter_data_outranged(self, df: pd.DataFrame, y_col: str) -> pd.DataFrame:
        """Filter data within hard min/max limit."""
        returnDf = df[(df[y_col] >= self.min_output_value) &
                      (df[y_col] <= self.max_output_value)]
        logDataframeChange(df, returnDf, self.logger,
                           f'filter_data_outranged({self.min_output_value}/{self.max_output_value}@{y_col})')
        return returnDf

    def get_score(self, y_true, y_pred) -> float:
        """Get the accuracy score."""
        pred_accuracy_score = 0
        if self.model_class == MODEL_TYPE_CLASSIFICATION:
            pred_accuracy_score = accuracy_score(y_true, y_pred)
        elif (self.model_class is None) or (self.model_class == MODEL_TYPE_REGRESSION):
            pred_accuracy_score = r2_score(y_true, y_pred)
        else:
            self.logger.error(
                f'Lost model class: {self.model_class} {type(self.model_class)} {self}')
            pred_accuracy_score = r2_score(y_true, y_pred)
        if isnan(pred_accuracy_score):
            self.logger.error(
                f'Accuracy score is nan: {self.model_class} {type(self.model_class)} {len(y_true)}:{len(y_pred)} {self}')
            pred_accuracy_score = 0
        return pred_accuracy_score
