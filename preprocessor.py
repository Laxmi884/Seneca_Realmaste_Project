
import datetime
from gc import garbage
from itertools import chain
import numpy as np
import pandas as pd
import re
from math import isnan
from pyrsistent import v

from base.util import allTypeToFloat, allTypeToInt, flattenList, stringToInt

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from base.base_cfg import BaseCfg
from base.const import NONE, RENT_PRICE_UPPER_LIMIT, SALE_PRICE_LOWER_LIMIT, UNKNOWN, DROP, MEAN, Mode
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from data.estimate_scale import PropertyType, PropertyTypeRegexp
from transformer.baseline import BaselineTransformer
from transformer.binary import BinaryTransformer
from transformer.baths import BthsTransformer
from transformer.bedrooms import RmsTransformer
from transformer.dates import DatesTransformer
from transformer.db_one_hot_array import DbOneHotArrayEncodingTransformer
from transformer.select_col import SelectColumnTransformer
from transformer.db_label import DbLabelTransformer
from transformer.db_numeric import DbNumericTransformer
from transformer.drop_row import DropRowTransformer
from transformer.const_label_map import getLevel, levelType, acType, \
    bsmtType, featType, constrType, garageType, lockerType, \
    heatType, fuelType, exposureType, laundryType, \
    parkingDesignationType, parkingFacilityType, balconyType, \
    ptpType
    
    
from transformer.simple_column import SimpleColumnTransformer
from transformer.street_n_st_num import StNumStTransformer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

logger = BaseCfg.getLogger(__name__)


def yearOfDateNumber(dateNumber, deltaDays=0):
    date = datetime.datetime.strptime(str(dateNumber), '%Y%m%d')
    date = date + datetime.timedelta(days=deltaDays)
    return date.year


def yearOfByField(row, field, deltaDays=0):
    return yearOfDateNumber(row[field], deltaDays)


def yearOf(field, deltaDays=0):
    return lambda row: yearOfByField(row, field, deltaDays)


def saletpSingleValue(row, saletp):
    if isinstance(saletp, str):
        return saletp
    elif isinstance(saletp, tuple) or isinstance(saletp, list):
        if len(saletp) == 1:
            return saletp[0]
        elif len(saletp) == 0:
            return 'Sale'
        if row['lpr'] is None or row['lpr'] == 0:
            return 'Sale'
        elif row['lp'] is None or row['lp'] == 0:
            return 'Lease'
    return None


def binarySaletpByRow(row, saletp):
    value = saletpSingleValue(row, saletp)
    if value == 'Sale':
        return 0
    elif value == 'Lease':
        return 1
    elif value is not None:
        logger.error('Unknown saletp value', value, row['_id'])
    return None


def ptype2SingleValue(row, ptype2):
    for value in ptype2:
        if not isinstance(value, str):
            continue
        if PropertyTypeRegexp.SEMI_DETACHED.match(value):
            return PropertyType.SEMI_DETACHED
        elif PropertyTypeRegexp.DETACHED.match(value):
            return PropertyType.DETACHED
        elif PropertyTypeRegexp.TOWNHOUSE.match(value):
            return PropertyType.TOWNHOUSE
        elif PropertyTypeRegexp.CONDO.match(value):
            return PropertyType.CONDO
    return None


def allTypeToFloatRow(_, value):
    return allTypeToFloat(value)


def allTypeToIntRow(_, value):
    return allTypeToInt(value)


def shallDrop(row):
    try:
        if (row['saletp-b'] == 0):
            if (row['lp-n'] == 0) | (row['lp'] is None) | (row['lp'] < SALE_PRICE_LOWER_LIMIT):
                return True
        if (row['saletp-b'] == 1):
            if (row['lpr-n'] == 0) | (row['lpr'] is None) | (row['lpr'] > RENT_PRICE_UPPER_LIMIT):
                return True
        if ('lst' in row.index) and (row['lst'] in ['Sld', 'Lsd']):
            if (row['sp-n'] == 0) | ('sp' not in row.index) | (row['sp'] is None):
                return True
    except Exception as e:
        logger.error(row)
        logger.error('shallDrop', e)
    return False


def taxYearRow(row, value):
    yr = allTypeToInt(value)
    if yr is not None:
        if 200 <= yr < 300:
            yr = yr % 100 + 2000
        if yr < 200:
            yr = yr + 2000
        if 1990 <= yr <= datetime.datetime.now().year:
            return yr
    return yearOfByField(row, 'onD', -183)


def laundryLevelRow(_, value):
    return getLevel(value)


def petsRow(_, value):
    value = str(value)[0]
    if value == 'Y':
        return 2
    elif value == 'R':
        return 1
    elif value == 'N':
        return 0
    else:
        return 2  # unknown


def balconyRow(_, value):
    return balconyType.get(value, 0)


SUFFIXES = {
    '-n': 'Number',
    '-c': 'Category Number',
    '-b': 'Binary 0/1',
    '-l': 'String Label',
}

###### our changes

# outliers replacer class
class OutlierRemover(BaseEstimator,TransformerMixin): # our own class to remove outliers - we will insert it to the pipeline 
    def __init__(self,factor=1.5):
        self.factor = factor # higher the factor, extreme would be the outliers removed.
        
    def outlier_detector(self,X,y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr)) 
        self.upper_bound.append(q3 + (self.factor * iqr))
        self.median.append(X.median()) # try to change

    def fit(self,X,y=None): # for each coulmn we will append corresponding boundary and the median value
        self.median = []
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self
    
    def transform(self,X,y=None): # then, with transform we will check is a value goes beyond the boundary, if so we replace it
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy() # change the copy
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = self.median[i] # replace outliers with the median
            X.iloc[:, i] = x # make the column copy

        return X # our transformed df
    
    
class Custom_Cat_Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy() # make the copy
        for col in X.columns: # for each column
            X[col] = X[col].interpolate(method='pad', limit_direction = "forward") # fffil
            X[col] = X[col].interpolate(method='bfill', limit_direction = "backward") # bffil
        return X # our transformed df


# Ontario, ... 3,4,5

class OneHotEncoderWithNames(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = Custom_Cat_Imputer() # 
        self.one_hot_encoder = OneHotEncoder()
        self.column_names = None
        
    def get_rep_value(self, X): # deal with wrongly placed numeric objects in the object feature
        mode = X.mode().tolist()[0]        
        X = X.apply(lambda x : mode if OneHotEncoderWithNames.to_n(x) == "!" else x) # ! if it was numeric
        return X
        
    @staticmethod
    def to_n(x):
        try:
            int(x)
            return "!" # if we can concert to int
        except ValueError: # if no
            return x
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        global one_hot_names # write the names of the encoded features
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X),columns = X.columns) # impute nulls
        X_imputed = X_imputed.apply(self.get_rep_value) # get rid of the "numeric" objects in feature
        X_one_hot_encoded = self.one_hot_encoder.fit_transform(X_imputed) # encode
        self.column_names = self.one_hot_encoder.get_feature_names(X_imputed.columns) # the encoded names
        X_df = pd.DataFrame(X_one_hot_encoded.toarray(), columns=self.column_names)
        one_hot_names = self.column_names
        return X_df


class custom_imputer(BaseEstimator, TransformerMixin): # based on the linear correlation between features
    def fit(self,X,y=None):
        self.corr_matrix = X.corr()
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for column in X.columns: # for each column
            col = list(self.corr_matrix[column]) # get the cor values
            vals = []
            for i,correlation in enumerate(col):
                vals.append((i,correlation)) 
            vals = sorted(vals,key = lambda y : y[1], reverse = True) # get the most correlated one
            for t in vals:
                if t[0] == i: # if it is us - ignore
                    continue
                else: # else, we have the max and break
                    val = t[0]
                    break
            med = X.iloc[:, val].median() # get the median of the correlated column
            X[column].fillna(med, inplace=True) # replace with the mdeian
        return X     
        
class custom_numeric_imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.models = [] 
        self.tests = {}
    def fit(self,X,y=None):
        features = set(X.columns.tolist()) 
        m = X.isna()
        select = []
        for col in list(features): # select only the cols with no nulls!
            if sum(m[col]) == 0:
                select.append(col)
                
        features = set(select) # these are our features 
        to_change = list(set(X.columns.tolist()).difference(features)) # columnns that do have at least one null
        
        for col in to_change: # for each null column
            ser = X[col]
            ser = ser.reset_index(drop=True) # we have to reset index...
            test = list(ser[ser.isnull()].index) # seperate nulls in our target
            train = list(ser[ser.notnull()].index) # seperate not nulls in our target
            """
            for our features get the non null target rows
            """
            X_train = X[list(features)].iloc[train, :] 
            y_train = X[[col]].iloc[train, :]
            model = LinearRegression() # regression
            model.fit(X_train,y_train) # train
            self.tests[col] = [model,test] # model, and test indicies
        else:
            self.features = list(features) # remembrt the list of features
        return self
            
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        for col in X.columns: # for each column
            data = self.tests.get(col,False)
            if not data: # if False - continue
                continue
            model,test_inds = data # unpack the model and the null indicies
            no_nulls = []
            """
            Select the null rows in the target to predict
            """
            preds = model.predict(X[self.features].iloc[test_inds, :])
            for cube in preds: # convert the result to one dim array
                no_nulls.append(cube[0])
                
            x = X.loc[:, col].copy()  # get the copy
            x[test_inds] = no_nulls # replace the null indicies with the predicted not nulls
            X.loc[:, col] = x # make the change
        return X    
    

# the dates in the proper dates format (not numeric)
class Dates_common_Pipeline(BaseEstimator,TransformerMixin): # convert the thing to the object format!
    def fit(self,X,y=None): 
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].interpolate(method='pad', limit_direction = "forward")#X[col].interpolate(method='linear')
            X[col] = X[col].interpolate(method='bfill', limit_direction = "backward")    
        X[X.columns.tolist()] = (X[X.columns.tolist()] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
        return X 
    

  # the numeric dates
class Dates_numeric_Pipeline(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): 
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].interpolate(method='linear').round(0)
            X[col] = X[col].interpolate(method='bfill', limit_direction = "backward")
        return X 
    
    

class Preprocessor(TransformerMixin, BaseEstimator):
    """ Transforms raw training and prediction data
    To build root transformer, use TRAIN mode and fit with full dataset(columns and rows). 
    To transform predict data, use PREDICT mode and fit with training/predict data.
    If columns changed, use PREDICT mode and fit with training data. It can transform both training and predict data.
    Different column sets need different preprocessors.

    Transforming steps:
    -. drop na columns if in training mode, return error if in prediction mode.
    -. convert binary saletp to 0 or 1, column name as 'saletp-b'
    -. convert ptype2 to single value. column name as 'ptype2-l'
    -. convert binary cols. column name as '-b'
    -. convert categorical columns to integers. column name as '-c'
    -. fill numeric columns to default values. column name as '-n'
    -. filter lat/lng to the range of [-180, 180] and drop null rows.
    """
    # binary use index as value, default 0
    cols_binary: dict = {
        'status':       ['U', 'A'],
        'den_fr':       ['N', 'Y'],
        'ens_lndry':    ['N', 'Y'],
        'cac_inc':      ['N', 'Y'],
        'comel_inc':    ['N', 'Y'],
        'heat_inc':     ['N', 'Y'],
        'prkg_inc':     ['N', 'Y'],
        'hydro_inc':    ['N', 'Y'],
        'water_inc':    ['N', 'Y'],
        'insur_bldg':   ['N', 'Y'],
        'tv':           ['N', 'Y'],
        'all_inc':      ['N', 'Y'],
        'furnished':    ['N', 'Y'],
        'retirement':   ['N', 'Y'],
        'pvt_ent':      ['N', 'Y'],
    }
    cols_label: dict = {  # na DROP means to remove the rows without label
        'lst':      {'na': UNKNOWN},
        'prov':     {'na': DROP},
        'area':     {'na': UNKNOWN},
        'city':     {'na': DROP},
        'cmty':     {'na': UNKNOWN},
        'st':       {'na': UNKNOWN},
        'zip':      {'na': UNKNOWN},
        'rltr':     {'na': UNKNOWN},
        #        'saletp':   {'na': DROP},
        'ptype2-l': {'na': DROP},
        'pstyl':    {'na': UNKNOWN},  # 131 types
        'ptp':      {'na': UNKNOWN},  # ptpType
        'zone':     {'na': UNKNOWN},
    }
    cols_array_label: dict = {
        'constr':   {'map': constrType, 'strType': False},
        'feat':     {'map': featType, 'strType': False},
        'bsmt':     {'map': bsmtType, 'strType': False},
        'fuel':     {'map': fuelType, 'strType': False},
        'laundry':  {'map': laundryType, 'strType': False},
        'park_desig': {'map': parkingDesignationType, 'strType': False},
        'ac':       {'map': acType, 'strType': True},
        'gatp':     {'map': garageType, 'strType': True},
        'lkr':      {'map': lockerType, 'strType': True},
        'heat':     {'map': heatType, 'strType': True},
        'fce':      {'map': exposureType, 'strType': True},
        'park_fac':  {'map': parkingFacilityType, 'strType': True},
    }
    cols_numeric: dict = {
        'lat':      {'na': DROP},
        'lng':      {'na': DROP},
        #        'st_num':   {'na': 0},
        'mfee':     {'na': 0},
        'tbdrms':   {'na': 0},
        'bdrms':    {'na': 0},
        'br_plus':  {'na': 0},
        'bthrms':   {'na': 0},
        'kch':      {'na': 0},
        'kch_plus': {'na': 0},
        'tgr':      {'na': 0},
        'gr':       {'na': 0},
        'lp':       {'na': 0},
        'lpr':      {'na': 0},
        'sp':       {'na': 0},
        'depth':    {'na': 0},
        'flt':      {'na': 0},
        # 'onD':      {'na': DROP},
        # 'offD':     {'na': 0},
    }

    # ----- Special cases: Done ---------
    cols_todo: dict = {
        'ptype':    {'na': 'r'},
    }

    # Done.
    cols_special: dict = {
        'lp':       {'na': DROP},
        'lpr':      {'na': DROP},
        'sp':       {'na': DROP},
        # extract number parts from street number
        'st_num':   {'to': 'st_num-n'},  # Done.
        'sqft':     {'to': 'sqft-n'},  # Done.
        'rmSqft':   {'to': 'sqft-n'},  # rmSqft or sqft estimator
        # from bltYr or rmBltYr or bltYr estimator
        'bltYr':    {'to': 'built_yr-n'},  # Done.
        'rmBltYr':  {'to': 'built_yr-n'},  # rmBltYr or bltYr estimator
        'ptype2':   {'to': 'ptype2-l'},  # ptype2. Done
        'ac':       {'to': 'ac-n'},  # ac
        'balcony':  {'to': 'balcony-n'},  # Done
        'laundry_lev': {'na': NONE},  # Done
        'pets':     {'na': UNKNOWN},  # Done
    }

    # Done
    cols_structured_todo: list[str] = [
        'rms',  # get primary bedroom dimensions and area, sum of all bedrooms deminsions, sum of all bedrooms area
        'bths',  # get bath numbers on each level => l0-l3 * number of bathrooms; l0-l3 * pices total
    ]

    cols_forsale_in_models: dict = {
        'tax':      {'na': MEAN},  # Done
        # the first half year counted as previous tax year
        # Done
        'taxyr':    {'na': lambda row: yearOfByField(row, 'onD', -183)},
    }
    cols_forsale_house_in_models: dict = {
        # MEAN of all depths when Detached/Semi-Detached/Freehold Townhouse
        'depth':    {'na': MEAN},  # Done
        # MEAN of all flt when Detached/Semi-Detached/Freehold Townhouse
        'flt':      {'na': MEAN},  # Done
    }
    cols_not_used: list[str] = [
        'la',  # la id : la.agnt[].id
        'la2',  # la2 id: la2.agnt[].id
        'schools',  # get 3 school names and ratings, rankings
    ]
    cols_condo: list[str] = [
        'unt'  # unit storey, total storey, percentage of total storey
    ]

    def __init__(
        self,
        collection_prefix: str = 'ml_',
        use_baseline: bool = True,
    ):
        self.collection_prefix = collection_prefix
        self.use_baseline = use_baseline

    def get_feature_columns(
        self,
        all_cols: list[str] = None,
    ) -> list[str]:
        """Get the feature columns for the model.

        Returns
        -------
        list[str]
            feature columns
        """
        if all_cols is None:
            if not hasattr(self.customTransformers) or self.customTransformers is None:
                raise ValueError('No transformer built yet')
        else:
            self.build_transformers(all_cols)
        cols = []
        for transformer in self.customTransformers:
            cols.append(transformer.get_feature_names_out())
        # return flatten(cols)
        return flattenList(cols)

    def build_transformers(self, all_cols): # data cleansing part
        """Build the transformers.
           The transformers need to work with less columns when the predictions has less data.

            Parameters
            ----------
            all_cols : list of strings
                all columns to transform

        """
        logger.info('Building transformers')
        all_cols = [*all_cols]

        colTransformerParams = [
            ('saletp-b', binarySaletpByRow, 'saletp', 'saletp-b'),
            ('ptype2-l', ptype2SingleValue, 'ptype2', 'ptype2-l', True),
        ]
        all_cols.append('saletp-b')
        all_cols.append('ptype2-l')
        # custom transformers
        if 'onD' in all_cols:
            datesTransformer = DatesTransformer(all_cols)
            colTransformerParams.append(
                ('onD', datesTransformer))
            all_cols.extend(datesTransformer.get_feature_names_out())
        if 'pets' in all_cols:
            colTransformerParams.append(('pets', petsRow, 'pets', 'pets-n'))
        if 'laundry_lev' in all_cols:
            colTransformerParams.append(
                ('laundry_lev', laundryLevelRow, 'laundry_lev', 'laundry_lev-n'))
        if 'balcony' in all_cols:
            colTransformerParams.append(
                ('balcony', balconyRow, 'balcony', 'balcony-n'))
        if 'flt' in all_cols:
            colTransformerParams.append(
                ('flt', allTypeToFloatRow, 'flt', 'flt-n'))
        if 'depth' in all_cols:
            colTransformerParams.append(
                ('depth', allTypeToFloatRow, 'depth', 'depth-n'))
        if 'tax' in all_cols:
            colTransformerParams.append(
                ('tax', allTypeToFloatRow, 'tax', 'tax-n'))
            colTransformerParams.append(
                ('taxyr', taxYearRow, 'taxyr', 'taxyr-n'))
        if ('bltYr' in all_cols) or ('rmBltYr' in all_cols):
            colTransformerParams.append(('bltYr', SelectColumnTransformer(
                new_col='bltYr-n', columns=['bltYr', 'rmBltYr'], func=stringToInt, as_na_value=None)))
        if ('sqft' in all_cols) or ('rmSqft' in all_cols):
            colTransformerParams.append(('sqft', SelectColumnTransformer(
                new_col='sqft-n', columns=['sqft', 'rmSqft'], func=stringToInt, as_na_value=None)))
        if 'st_num' in all_cols:
            colTransformerParams.append(
                ('st_num', allTypeToIntRow, 'st_num', 'st_num-n'))
        # array labels
        for k, v in self.cols_array_label.items():
            if k in all_cols:
                if v['strType']:
                    transformer = DbOneHotArrayEncodingTransformer(
                        col=k,
                        map=v['map'],
                        sufix='-b',
                        na_value=None,
                        collection=self.label_collection,
                    )
                else:
                    transformer = DbOneHotArrayEncodingTransformer(
                        col=k,
                        map=v['map'],
                        sufix='-b',
                        # na_value=None,
                        # collection=self.label_collection,
                    )
                colTransformerParams.append((f'{k}_x', transformer))
                all_cols.extend(transformer.get_feature_names_out())
        # binary columns
        for k, v in self.cols_binary.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}-b', BinaryTransformer(v, k), k, f'{k}-b'))
                all_cols.append(f'{k}-b')
        # categorical columns
        for k, v in self.cols_label.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}-c', DbLabelTransformer(
                        col=k,
                        na_value=v['na'],
                        collection=self.label_collection,
                    ), k, f'{k}-c'))
                all_cols.append(f'{k}-c')
        # numerical columns
        for k, v in self.cols_numeric.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}-n', DbNumericTransformer(
                        self.number_collection,
                        col=k,
                        na_value=v['na'],), k, f'{k}-n'))
                all_cols.append(f'{k}-n')
        if 'st_num' in all_cols:
            stNumStTransformer = StNumStTransformer()
            colTransformerParams.append(
                ('st_num-st', stNumStTransformer))
            all_cols.extend(stNumStTransformer.get_feature_names_out())
        # rms and bths
        if 'rms' in all_cols:
            colTransformerParams.append(('rms', RmsTransformer()))
        if 'bths' in all_cols:
            colTransformerParams.append(('bths', BthsTransformer()))
        # drop rows: this operation has to be done outside of the pipeline
        # the drop operation is dependent on the model's usage of columns
        drop_na_cols = []
        for k, v in chain(self.cols_label.items(), self.cols_numeric.items()):
            if (k in all_cols) and (v['na'] is DROP):
                drop_na_cols.append(k)
        self.drop_na_cols_ = drop_na_cols
        colTransformerParams.append(
            ('drop_na', DropRowTransformer(drop_cols=drop_na_cols))
        )
        colTransformerParams.append(
            ('drop_check', DropRowTransformer(drop_func=shallDrop))
        )

        # create the pipeline
        self.customTransformers = []
        self.customTransformers.append(SimpleColumnTransformer(
            colTransformerParams))
        if self.use_baseline:
            # baseline transformer, which run on the transformed data from previous step
            self.customTransformers.append(
                SimpleColumnTransformer([('baseline', BaselineTransformer(
                    sale=None,
                    collection=self.baseline_collection))]
                )
            )

        return self.customTransformers

    def fit(self, Xdf: pd.DataFrame, y=None): # Xdf is raw data from the database
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
        self.label_collection = self.collection_prefix + 'label'
        self.number_collection = self.collection_prefix + 'number'
        self.baseline_collection = self.collection_prefix + 'baseline'

        self.build_transformers(Xdf.columns)
        # fit the first transformer only
        self.customTransformers[0].fit(Xdf, y)
        self.n_features_ = Xdf.shape[1]
        if len(self.customTransformers) == 1:
            self.fited_all_ = True
        else:
            self.fited_all_ = False
        return self

    def transform(self, Xdf: pd.DataFrame):
        """Transform the dataframe to the baseline format.

            Parameters
            ----------
            df : pd.DataFrame
                the dataframe to transform

        """
        if self.n_features_ is None:
            raise ValueError('The transformer has not been fitted yet.')

        # if self.customTransformers is None:
        #     self.build_transformers(Xdf.columns)
        #     self.customTransformers.fit(Xdf)
        logger.info('Transforming')
        #pd.set_option('mode.chained_assignment', None)
        Xdf = self.customTransformers[0].transform(Xdf)
        self.Xdf = Xdf
        if len(self.customTransformers) > 1:
            # fit the second transformer
            for i in range(1, len(self.customTransformers)):
                if self.fited_all_ is False:
                    self.customTransformers[i].fit(Xdf)
                self.Xdf = Xdf  # for debug
                logger.debug(f'before transform {i}: {Xdf.shape}')
                logger.debug(Xdf.head())
                Xdf = self.customTransformers[i].transform(Xdf)
                logger.debug(f'after transform {i}: {Xdf.shape}')
                logger.debug(Xdf.head())
                self.Xdf = Xdf  # for debug
            self.fited_all_ = True
 
        ###### Prepocessing part (some of it)
        
        threshold = 0.5
        
        na_percentages = Xdf.isna().sum() / Xdf.shape[0]
        cols_to_drop = list(na_percentages[na_percentages > threshold].index)
        
        
        # be sure that the targets are not dropped
        
        if "bltYr-n" in cols_to_drop:
            cols_to_drop.remove("bltYr-n")
        if "sqft-n" in cols_to_drop:
            cols_to_drop.remove("sqft-n")                
        if "bltYr-n" in cols_to_drop:
            cols_to_drop.remove("sp-n")
        Xdf = Xdf.drop(cols_to_drop, axis=1)
        
        nested_cols = Xdf.applymap(type).isin([dict, list]).any() # dropping the nested things
        Xdf = Xdf.loc[:, ~nested_cols]
        
        existing = list(Xdf.columns)
        
        dates_special = ["offD","offD-month-n","offD-season-n","offD-year-n","onD","onD-month-n","onD-season-n","onD-week-n","onD-year-n","bltYr-n","rmBltYr", "taxyr-n",\
                         "taxyr"]
            
        dates_special = list(set(existing).intersection(set(dates_special))) # be sure that the numeric date exist
        
        num_cols = [col for col in Xdf.columns if (Xdf.dtypes[col] in ["int64","int32","float64","float32"] and col not in dates_special)]  
        ob = [col for col in Xdf.columns if col not in num_cols and col not in dates_special and Xdf.dtypes[col] != "datetime64[ns]"]  # the text features
        
        common_dates = [col for col in Xdf.columns if col not in num_cols and col not in ob and col not in dates_special] # the proper dates (not numeric)
        #dates_special = dates_special + [col for col in Xdf.columns if col not in num_cols and col not in ob and col not in dates_special]
        
        encoders = []
        others = []

        for col in Xdf[ob].columns: # {N,U,M}
            lst = list(set(Xdf[ob][col]))
            t = len(lst)
            
            if np.nan in lst or None in lst: 
                t -= 1
            if 2 <= t <= 17 and col not in [ # if the length of the unique set of values for feature is from 2 to 17
            'saletp-b', 'ptype2-l',          # and not the index one
            'prov', 'area', 'city',
            '_id',
        ]:
                encoders.append(col)
            else:
                others.append(col)
        
        numeric = Pipeline([ 
        ('imputer', custom_numeric_imputer()), # regression class
        ("outliers_removal", OutlierRemover())])  # outliers and no scaling here
        
        dates_pipe_spec = Pipeline([('numeric_dates', Dates_numeric_Pipeline())]) # interpolate na
        dates_pipe_common = Pipeline([('rest_dates', Dates_common_Pipeline())]) # bfil and ffil for na
        
        str_pipe_encoders = Pipeline([("one_hot_imputer", OneHotEncoderWithNames())]) # encoders
        str_pipe_others = Pipeline([('imputer', SimpleImputer(strategy="most_frequent"))]) # with the mode
        
        full_pipeline = ColumnTransformer([("num", numeric, num_cols), ("numeric_dates",dates_pipe_spec,dates_special),("common_dates",dates_pipe_common,common_dates) ,("str_encode",str_pipe_encoders,encoders),("str_others",str_pipe_others,others)])

        g = full_pipeline.fit_transform(Xdf)
        
        columns = num_cols + dates_special+ common_dates +list(one_hot_names) + others # match the order
        
        
        z = pd.DataFrame(g,columns=columns)
        
        z[common_dates] = z[common_dates].apply(lambda x : pd.to_datetime(x , unit='s'), axis = 1) # covert the proper dates from numeric to date format again
        
        dates_to_change = dates_special
        
        #for c in dates_special:
            #if Xdf[dates_special].dtypes[c] != "datetime64[ns]":
                #dates_to_change.append(c)
                
        all_cols = num_cols+dates_to_change+list(one_hot_names)
        z[all_cols] = z[all_cols].apply(pd.to_numeric) #convert all numeric from onbject to float again
        z = z.reindex(sorted(z.columns), axis=1)
        Xdf = z
        Xdf.head(n=100).to_excel("preporcesed_data.xlsx")
        self.encoded_hot = num_cols+list(one_hot_names) # add the rest of the cols
        self.Xdf = Xdf
        #print(Xdf.info())
        return Xdf
    
    
    
    
    
    
