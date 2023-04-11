# Mapping the labels to the corresponding numbers.
# Save the mapping to database and reuse when needed.
from base.base_cfg import BaseCfg
from base.mongo import getMongoClient
from base.timer import Timer
from base.const import SAVE_LABEL_TO_DB, Mode
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

logger = BaseCfg.getLogger(__name__)


class DbLabelTransformer(TransformerMixin, BaseEstimator):
    """Transforms labels to integers and save the mapping to database

    Parameters
    ----------
    collection : mongodb collection
        The mongodb collection to save the mapping to.
    col: str
        The feature name to save the mapping as.
    mode: Mode
        The mode of the transformer. PREDICT or TRAIN.
    na_value: str
        The value to fill the empty fields with.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
        Right now, this is always 1.
    labels_ : dict
        A dictionary mapping the column label names to the label indexes.
    labels_index_ : dict
        A dictionary mapping the column label indexes to the label names.
    labels_index_next_ : int
        The next label index to use.
    """

    def __init__(
        self,
        col: str,
        na_value=None,
        collection: str = None,
    ):
        self.collection = collection
        self.col = col
        self.na_value = na_value

    def get_feature_names_out(self):
        return [self.col+'-c']

    def set_params(self, **params):
        ret = super().set_params(**params)
        self._connect_db()
        return ret

    def _connect_db(self):
        if hasattr(self, '_db_connected') and self._db_connected:
            return getMongoClient()
        if not self.collection:
            logger.warn("No collection specified")
            return
        # create index on col
        MongoDB = getMongoClient()
        if not MongoDB.hasIndex(self.collection, 'col_1_i_1'):
            MongoDB.createIndex(self.collection, fields=[
                ("col", 1), ('i', 1)], unique=True)
        self._db_connected = True
        return MongoDB

    def load_from_db(self):
        if self.collection is None:
            return
        MongoDB = self._connect_db()
        self.labels_ = {}
        self.labels_index_ = {}
        self.labels_index_next_ = 0
        for doc in MongoDB.find(self.collection, {'col': self.col}):
            self.labels_[doc['label']] = doc['i']
            self.labels_index_[doc['i']] = doc['label']
            if doc['i'] > self.labels_index_next_:
                self.labels_index_next_ = doc['i']

    def save_to_db(self):
        if self.collection is None:
            return
        to_save = []
        for label, index in self.labels_.items():
            to_save.append({
                "_id": f"{self.col}:{label}",
                "col": self.col,
                "label": label,
                "i": index,
                "cnt": self.labels_count_[label],
            })
        MongoDB = self._connect_db()
        MongoDB.deleteMany(self.collection, {'col': self.col})
        MongoDB.insertMany(self.collection, to_save)

    def _save_mapping(self, col: str, label: str, index: int, count: int):
        if self.collection is None:
            return
        MongoDB = self._connect_db()
        MongoDB.save(self.collection, {
            "_id": f"{col}:{label}",
            "col": col,
            "label": label,
            "i": index,
            "cnt": count,
        })

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
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
        # logger.debug(
        #    f'label {self.col} fit')
        # X = check_array(X, accept_sparse=True)

        # self.n_features_ = X.shape[1]

        t = Timer(self.col, logger)

        # logger.debug(f'set self.labels_ {self.col}')
        self.labels_ = {}
        self.labels_count_ = {}
        self.labels_index_ = {}
        self.labels_index_next_ = 0
        try:
            col_labels = X.value_counts()
            col_labels.index = col_labels.index.astype(str)
            col_labels = col_labels.sort_values(
                ascending=False)  # sort by count, more to less
        except TypeError as e:
            logger.fatal(X)
            logger.fatal(e)
            raise e
        # make sure the index is string
        totalLabels = 0
        for label, count in col_labels.items():
            if label is None:
                if self.na_value is not None:
                    label = self.na_value
                else:
                    continue
            cur_index = self.labels_index_next_
            self.labels_[label] = cur_index
            self.labels_index_[cur_index] = label
            self.labels_index_next_ += 1
            self.labels_count_[label] = count
            totalLabels += 1
            # print(self.labels_)
        self.save_to_db()
        logger.info(f'{self.col} fit labels:{totalLabels}')
        t.stop(X.shape[0])

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        # check_is_fitted(self, 'n_features_')

        # # Input validation
        # X = check_array(X, accept_sparse=True)

        # # Check that the input is of the same shape as the one passed
        # # during fit.
        # if X.shape[1] != self.n_features_:
        #     raise ValueError('Shape of input is different from what was seen'
        #                      'in `fit`')
        # we may need to use a wrapper here
        # X = pd.DataFrame(X)
        # logger.debug(f'label {self.col} transform')
        # fill na_value
        if self.na_value is not None:
            X.fillna(value=self.na_value, inplace=True)

        # transform with new column names and add new labels as needed

        def map_value(label, self):
            if label is None:
                return None
            if isinstance(label, list):  # when label is a list
                if len(label) == 0:
                    return None
                return max([map_value(l, self) for l in label])
            index = self.labels_.get(label, None)
            if isinstance(index, int):
                return index
            logger.info(
                f'unknown label {label} {type(label)} in column {self.col}')
            label = str(label)
            index = self.labels_index_next_
            self.labels_[label] = index
            self.labels_index_next_ += 1
            self.labels_count_[label] = 1
            self._save_mapping(self.col, label, index, 1)
            return index
        X = X.apply(
            lambda val: map_value(val, self))
        # return X
        return X
