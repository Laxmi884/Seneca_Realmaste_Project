
from math import isnan
from base.base_cfg import BaseCfg

from base.timer import Timer
from base.util import flattenList, isNanOrNone
from numpy import nan
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.db_label import DbLabelTransformer
from base.const import SAVE_LABEL_TO_DB, Mode

logger = BaseCfg.getLogger(__name__)

# TODO: mapping all combinations of array values to one output column as int.


class DbOneHotArrayEncodingTransformer(DbLabelTransformer):
    """One hot encoding for field of string array.

    Parameters
    ----------
    col: str
        the source column
    map: dict
        the mapping from original string to new feature string. It may be an N to 1 or 1 to N mapping.
    collection : mongodb collection or None
        The mongodb collection to save the mapping to. When a collection name is provided, transformer will generate a category column ended -c.
    col: str
        The feature name to save the mapping as.
    mode: Mode
        The mode of the transformer. PREDICT or TRAIN.
    na_value: str
        The value to fill the empty fields with.
    """

    def __init__(
        self,
        col: str,
        map: dict = None,
        sufix: str = '-b',
        na_value=None,
        collection: str = None,
    ):
        super().__init__(col, na_value, collection, )
        self.sufix = sufix
        self.map = map

    def get_feature_names_out(self):
        if hasattr(self, '_target_cols') and self._target_cols is not None:
            return self._target_cols
        retSet = set()
        # category column
        self.col_category = self.col+'-c'
        retSet.add(self.col_category)
        # feature columns
        if self.map is not None:
            self.map_ = {}
            for (x, y) in self.map.items():
                if isinstance(y, list):
                    x_target_cols = []
                    for v in y:
                        x_target_cols.append(self._new_col_name(v))
                    self.map_[x] = x_target_cols
                elif isinstance(y, str):
                    self.map_[x] = self._new_col_name(y)
                else:
                    raise ValueError(f'{y} is not a string or list')
        if getattr(self, 'map_') is None:
            raise ValueError('map is not set')
        for v in self.map_.values():
            if isinstance(v, list):
                for vv in v:
                    retSet.add(vv)
            elif isinstance(v, str):
                retSet.add(v)
        self._target_cols = list(retSet)
        return self._target_cols

    def set_params(self, **params):
        ret = super().set_params(**params)
        self._target_cols = None
        return ret

    def _new_col_name(self, col_name):
        if col_name.startswith(self.col):
            return col_name + self.sufix
        else:
            return self.col + '-' + col_name + self.sufix

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        # logger.debug(f'fit {self.col}')
        self._errors = {}

        if self.map is None:
            # logger.debug(f'fit {self.col} build map')
            col_index = X[self.col].value_counts().index
            # TODO: support array?
            col_values = [v + self.sufix for v in col_index]
            self.map_ = dict(zip(col_index, col_values))

        Xs = X[self.col].explode(ignore_index=False).apply(lambda x: self.map_.get(
            x, x)).explode(ignore_index=False)
        super().fit(Xs, y)
        # print('self getattr labels-', getattr(self, 'labels-', 'None'))
        # print(f'self.col {self.col} in self.labels-',
        #       (self.col not in self.labels_))

        self._target_cols = None
        self.n_cols_ = len(self.get_feature_names_out())
        return self

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : {array-like, sparse matrix}, shape (n_samples, n_features_)
            The transformed data.
        """
        # logger.debug(f'transform {self.col}')
        t = Timer(self.col, logger)
        list_value_count = 0
        str_value_count = 0
        error_count = 0
        default_col_name = self.map.get('-', None)
        new_cols = self.get_feature_names_out()
        cols_to_add = [col for col in new_cols if col not in X.columns]
        X[cols_to_add] = 0

        def _set_value(value, row):
            nonlocal error_count
            col_name = self.map_.get(value, default_col_name)
            if isinstance(col_name, list):
                for col_name_item in col_name:
                    #X.loc[i, (self.col+'-'+col_name_item)] = 1
                    row[col_name_item] = 1
            elif isinstance(col_name, str):
                #X.loc[i, (self.col+'-'+col_name)] = 1
                row[col_name] = 1
            elif isNanOrNone(value):
                pass
            else:
                error_count += 1
                if self._errors.get(value, None) is not None:
                    self._errors[value] += 1
                    return
                self._errors[value] = 1
                logger.error(f'{self.col} {value} {type(value)} not in map')

        if self.col in X.columns:
            # set value column(s)
            def _transform(row):
                nonlocal list_value_count, str_value_count, error_count, self
                value = row[self.col]
                if isinstance(value, list):
                    for v in value:
                        _set_value(v, row)
                        list_value_count += 1
                elif isinstance(value, str) or isinstance(value, int):
                    # sometimes the value is int
                    _set_value(value, row)
                    str_value_count += 1
                elif isNanOrNone(value):
                    pass
                else:
                    logger.error(
                        f'{self.col} {value} {type(value)} not in map')
                    error_count += 1
                return row
            X = X.apply(_transform, axis=1)

            # set category column
            if self.col_category is not None:
                def _get_category(value):
                    if isinstance(value, list):
                        ret = list(set(flattenList([self.map_.get(v, v) for v in value])))
                        ret.sort()
                        return ret
                    elif isinstance(value, str):
                        return self.map_.get(value, value)
                    elif isNanOrNone(value):
                        return None
                    else:
                        logger.error(
                            f'{self.col} {value} {type(value)} not in map for category {self.col_category}')
                        return None
                Xs = X[self.col].apply(_get_category)
                X[self.col_category] = super().transform(Xs)
        else:
            logger.error(f'{self.col} not in X')

        logger.info(
            f'{self.col} list:{list_value_count} str:{str_value_count} error:{error_count}')
        t.stop(X.shape[0])
        return X
