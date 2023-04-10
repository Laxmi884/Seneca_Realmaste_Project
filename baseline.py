
from datetime import datetime
from math import isnan
import re
import pandas as pd
import numpy as np
from base.const import DROP
from base.base_cfg import BaseCfg
from base.mongo import getMongoClient
from base.timer import Timer
from base.util import dateFromNum, printColumns
from sklearn.base import BaseEstimator, TransformerMixin
from transformer.const_label_map import getLevel

from transformer.prep import deal_with_sqft
from transformer.prep import Outliers_removal_ml_dif_rand 
from transformer.prep import OutlierRemover_general_iqr
from transformer.prep import null_imputer_ml
from transformer.prep import Custom_Cat_Imputer
from transformer.prep import OneHotEncoderWithNames
from transformer.prep import one_hot_names
from transformer.prep import OutlierRemover_distrs
from transformer.prep import Outlier_Remover_with_Z
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


logger = BaseCfg.getLogger(__name__)
FEATURES_KEY = 'features'
VALUES_KEY = 'values'
RATIOS_KEY = 'ratios'
SALETP_COL = 'saletp-b'

EXT_N = '-bl-n'
EXT_M = '-blm-n'


def findColumnName(df, col):
    for ext in ['-n', '-b', '-l', '-e-n', '-e', '']:
        if f"{col}{ext}" in df.columns:
            return f"{col}{ext}"
    for colInDf in df.columns:
        if colInDf.startswith(col) and col.endswith('-n'):
            return colInDf
    return None


class BaselineTransformer(BaseEstimator, TransformerMixin):
    """baseline transformer is a statistics collector and transformer.
    transform baseline features to dollar measurable numeric values.

    Parameters
    ----------
    None
    """
    discrete_cols = ['cmty', 'gr', 'bdrms', 'br_plus',
                     'bthrms', 'kch', 'onD-month', ]
    # continous columns are also for sale properties only. rent properties do not have these columns
    continous_cols = ['flt', 'depth', 'tax', 'mfee', 'bltYr', 'sqft', ]
    percentiles = [0.05, 0.32, 0.5, 0.68, 0.95]

    def __init__(
        self,
        sale: bool = None,
        collection: str = None,
    ):
        self.sale = sale  # True for sale, False for rent, None for both
        # when collection is not None, save the stats to the collection
        self.collection = collection
        pass

    def get_feature_names_out(self):
        self.discrete_new_cols = [f"{col}{EXT_N}" for col in self.discrete_cols] + \
            [f"{col}{EXT_M}" for col in self.discrete_cols]
        # for rent
        if self.sale == False:
            return self.discrete_new_cols

        # for sale
        self.continous_new_cols = [f"{col}{EXT_N}" for col in self.continous_cols] + \
            [f"{col}{EXT_M}" for col in self.continous_cols]
        return self.discrete_new_cols + self.continous_new_cols

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : dataframe
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        timer = Timer('baseline', logger)
        timer.start()
        totalCount = 0
        self.stats_ = [{}, {}]
        SP_N = 'sp-n'
        X = X.loc[X['ptype2-l'] != DROP]
        # X = X.where(X[SP_N] > 0).dropna()
        XX = [None, None]
        XX[0] = X.loc[(X[SALETP_COL] == 0)]
        XX[0] = XX[0].fillna({SP_N: XX[0]['lpr-n']})
        XX[0] = XX[0].loc[XX[0][SP_N] > 0]
        if self.sale is None or self.sale == True:
            XX[1] = X.loc[(X[SALETP_COL] == 1)]
            XX[1] = XX[1].fillna({SP_N: XX[1]['lpr-n']})
            XX[1] = XX[1].loc[XX[1][SP_N] > 0]

        for i in [0, 1]:
            if XX[i] is None:
                continue
            self.stats(self.stats_[i], XX[i], SP_N, i)
        if self.collection is not None:
            self.saveStats()
        timer.stop(totalCount)
        return self

    def saveStats(self):
        totalRecords = 0
        if self.collection is None:
            logger.warning('No collection to save to')
            return totalRecords
        mongo = getMongoClient()
        for i in range(2):
            stats = self.stats_[i]
            YYMM = stats['saveKey']
            saletp = 's' if i == 0 else 'r'
            keyPrefix = f"{YYMM}{saletp}"
            record = {
                '_id': keyPrefix,
                'ts': datetime.now(),
                'count': stats['count'],
                'seconds': stats['seconds'],
                'dateSpan': stats['dateSpan'],
                'startOnD': stats['startOnD'],
                'endOnD': stats['endOnD'],
            }
            totalRecords += 1
            mongo.save(self.collection, record)
            for key in stats:
                if isinstance(key, tuple):
                    if key[2] == DROP:
                        continue
                    cityTypeStats = stats[key]
                    record = {
                        '_id': f"{keyPrefix}:{key[0]}:{key[1]}:{key[2]}",
                        'count': cityTypeStats['count'],
                        'mean': cityTypeStats['mean'],
                        'std': cityTypeStats['std'],
                        'min': cityTypeStats['min'],
                        '5%': cityTypeStats['5%'],
                        '32%': cityTypeStats['32%'],
                        '50%': cityTypeStats['50%'],
                        '68%': cityTypeStats['68%'],
                        '95%': cityTypeStats['95%'],
                        'max': cityTypeStats['max'],
                    }
                    totalRecords += 1
                    mongo.save(self.collection, record)
                    for feature in cityTypeStats[FEATURES_KEY]:
                        featureStats = cityTypeStats[FEATURES_KEY][feature]
                        record = {
                            '_id': f"{keyPrefix}:{key[0]}:{key[1]}:{key[2]}:{feature}"}
                        if isinstance(featureStats.get(VALUES_KEY, None), dict):
                            record[VALUES_KEY] = {}
                            # logger.debug(
                            #     f"featureStats[VALUES_KEY]: {featureStats[VALUES_KEY]}")
                            for value in featureStats[VALUES_KEY]:
                                record[VALUES_KEY][f'{value}'] = featureStats[VALUES_KEY][value]
                            for fkey in featureStats:
                                if fkey != VALUES_KEY:
                                    record[fkey] = featureStats[fkey]
                        else:
                            record = record | featureStats
                        totalRecords += 1
                        try:
                            mongo.save(self.collection, record)
                        except Exception as e:
                            logger.error(e)
                            logger.error(record)
                            logger.error(featureStats.get(
                                VALUES_KEY, 'Not found'))
                            logger.error(
                                type(featureStats.get(VALUES_KEY, None)))

        return totalRecords

    def stats(self, stats_, X, SP_N, saletp):
        stats_['count'] = X.shape[0]
        startTs = datetime.now()
        dfOverall = X.groupby(
            ['prov', 'city', 'ptype2-l'])[SP_N].describe(percentiles=self.percentiles)
        # logger.info(
        #     f"do stats: {[col for col in X.columns if 'sp' in col]}")
        # logger.info(f"do stats: {dfOverall.head()}")
        # logger.info(X[['sp', SP_N]].head(100))
        for index, row in dfOverall.iterrows():
            key = (index[0], index[1], index[2])  # prov-city-ptype2-l
            dict = row.to_dict()
            dict[FEATURES_KEY] = {}
            stats_[key] = dict
            # logger.info(f"stats: {key} {dict}")
        for col in self.discrete_cols:
            col_n = self.featureOverallStats(stats_, SP_N, X, col)
            if col_n is None:
                continue
            self.discreteStats(stats_, X, SP_N, col, col_n)
        if saletp == 0:  # rent properties do not have continous columns
            for col in self.continous_cols:
                col_n = self.featureOverallStats(stats_, SP_N, X, col)
                if col_n is None:
                    continue
                self.continousStats(stats_, X, SP_N, col, col_n)
        onDStats = X['onD'].agg(['min', 'max'])
        endOnD = dateFromNum(onDStats['max'])
        startOnD = dateFromNum(onDStats['min'])
        dateSpan = (endOnD - startOnD).days
        stats_['dateSpan'] = dateSpan
        stats_['startOnD'] = startOnD.strftime('%Y-%m-%d')
        stats_['endOnD'] = endOnD.strftime('%Y-%m-%d')
        stats_['saveKey'] = endOnD.strftime('%Y%m')
        endTs = datetime.now()
        seconds = (endTs - startTs).total_seconds()
        stats_['seconds'] = seconds

    def featureOverallStats(self, stats_, SP_N, X, col):
        col_n = findColumnName(X, col)
        if col_n is None:
            logger.warning(f'column {col} not found')
            return col_n
        df = X.groupby(
            ['prov', 'city', 'ptype2-l'])[col_n].describe(percentiles=self.percentiles)
        for index, row in df.iterrows():
            # prov-city-ptype2-l
            key = (index[0], index[1], index[2])
            featureStats = row.to_dict()
            featureStats['col'] = col_n
            # stats for each feature
            stats_[key][FEATURES_KEY][col] = featureStats
            # logger.info(f"{col} {key} {featureStats}")
        return col_n

    def discreteStats(self, stats_, X, SP_N, col, col_n):
        df = X.groupby(
            ['prov', 'city', 'ptype2-l', col_n])[SP_N].describe(percentiles=self.percentiles)
        for index, row in df.iterrows():
            if index[2] == DROP:
                continue
            # prov-city-ptype2-l
            key = (index[0], index[1], index[2])
            stats = stats_[key]
            featureStats = stats[FEATURES_KEY][col]
            featureValueStats = row.to_dict()
            if VALUES_KEY not in featureStats:
                featureStats[VALUES_KEY] = {}
            # delta of value to mean
            feature_mean = featureStats.get('mean', stats.get('mean', 0))
            feature_median = featureStats.get('50%', stats.get('50%', 0))
            feature_value_mean = featureValueStats['mean']
            feature_value_median = featureValueStats['50%']
            delta = feature_value_mean - feature_mean
            # delta of value to median
            deltaM = feature_value_median - feature_median
            # debug
            # logger.debug(
            #     f"discreteStats {col} {key} {feature_mean} {feature_median} {featureStats}")
            # price/delta ratio by mean
            if delta == 0:
                ratio = 0
            else:
                ratio = (
                    feature_value_mean - stats['mean']) / delta
            # price/delta ratio by median
            if deltaM == 0:
                ratioM = 0
            else:
                ratioM = (
                    feature_value_median - stats['50%']) / deltaM
            featureValueStats['delta'] = delta
            featureValueStats['deltaM'] = deltaM
            featureValueStats['ratio'] = ratio
            featureValueStats['ratioM'] = ratioM
            featureStats[VALUES_KEY][index[3]] = featureValueStats

    def continousStats(self, stats_, X, SP_N, col, col_n):
        # calculate delta ratio
        col_n_delta = f"{col}-delta"
        col_n_sp_diff = f"{col}-sp-diff"
        col_n_ratio = f"{col}-ratio"
        X[[col_n_delta, col_n_sp_diff, col_n_ratio]] = 0

        def _calcRatio(row):
            _stats = stats_.get(
                (row['prov'], row['city'], row['ptype2-l']), None)
            if _stats is None:
                logger.warning(
                    f"stats not found for {row['prov']} {row['city']} {row['ptype2-l']}")
                return row
            # 1. calculate value delta: delta = value - mean
            delta = row[col_n] - _stats[FEATURES_KEY][col]['mean']
            row[col_n_delta] = delta
            # 2. calculate soldprice difference
            sp_diff = row[SP_N] - _stats['mean']
            row[col_n_sp_diff] = sp_diff
            # 3. calculate ratio
            if delta != 0:
                row[col_n_ratio] = sp_diff / delta
            # logger.debug(
            #     f"ratio {col} {col_n} {row[col_n_ratio]} {sp_diff} / {delta}({row[col_n]} - {_stats[FEATURES_KEY][col]['mean']})")
            return row
        X = X.apply(_calcRatio, axis=1)
        self.X = X  # for debug
        # calculate delta ratio stats
        # remove outliers
        q1 = X[col_n_ratio].quantile(0.25)
        q3 = X[col_n_ratio].quantile(0.75)
        iqrange = q3 - q1
        lower_bound = q1 - (1.5 * iqrange)
        upper_bound = q3 + (1.5 * iqrange)
        X = X[(X[col_n_ratio] > lower_bound) & (X[col_n_ratio] < upper_bound)]
        # df = X.groupby(
        #    ['prov', 'city', 'ptype2-l'])[col_n_ratio].describe(include=np.number, percentiles=self.percentiles)
        df = X.groupby(
            ['prov', 'city', 'ptype2-l'])[col_n_ratio].aggregate(['mean', 'median', 'min', 'max', 'std', 'count'])
        # logger.info(f"{col} ratio stats {df.head()}")
        df['50%'] = df['median']
        for index, row in df.iterrows():
            if index[2] == DROP:
                continue
            # prov-city-ptype2-l
            key = (index[0], index[1], index[2])
            featureStats = row.to_dict()
            # stats for each feature
            stats_[key][FEATURES_KEY][col][RATIOS_KEY] = featureStats

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : dataframe. The input samples.

        Returns
        -------
        X_transformed : dataframe. The transformed data.
        """

        # logger.debug(f'transform rms')
        timer = Timer('baseline', logger)
        timer.start()
        
        ######################
        feats_suf = []
        sufs = ['-b', '-n', '-c', '-l']  #bths-n, bths  

        for col1 in X.columns:
            if col1 in  ["ptype2-l"]:
                continue
            for s in sufs:
                if col1.endswith(s):    
                    feats_suf.append(col1)
        feats_suf  = list(set(feats_suf)) 
        
        
        g = X[feats_suf]
        years = [col for col in g.columns if (col.startswith("onD") or col.startswith("offD"))]
        
        threshold = 1
        na_percentages = g.isna().sum() / g.shape[0]
        cols_to_drop = list(na_percentages[na_percentages >= threshold].index)

        if "bltYr-n" in cols_to_drop:
            cols_to_drop.remove("bltYr-n")
        if "sqft-n" in cols_to_drop:
            cols_to_drop.remove("sqft-n")                
        if "bltYr-n" in cols_to_drop:
            cols_to_drop.remove("sp-n")
        
            
        g= g.drop(cols_to_drop, axis=1)    
        
        feats_suf = list(g.columns)
        
        
        encodes = []
        num = []
        for col in g.columns:
            if np.nanmax(g[col]) <= 50 and np.sum(g[col]) % 1 == 0:
                encodes.append(col)
            else:
                num.append(col)
                
        nums = Pipeline([ 
            ('imputer_num', null_imputer_ml(dec=True))]) 
                
        cats = Pipeline([ 
            ('imputer_cat', Custom_Cat_Imputer())]) 
        

        g[encodes] = cats.fit_transform(g[encodes])
        g = nums.fit_transform(g)
        

                
        #OutlierRemover_general_iqr
        #Outlier_Remover_with_Z
        #Outliers_removal_ml_dif_rand
        # OutlierRemover_distrs
        
        outs = OutlierRemover_distrs(cols=["lng-n","lat-n"]+years+["taxyr-n","rmBltYr","bltYr-n"])
        
        g[num] = outs.fit_transform(g[num])
        
        X[feats_suf] = g[feats_suf]
        
        ######################
        
        totalCount = 0
        transformedCount = 0
        new_cols = self.get_feature_names_out()
        X.loc[:, new_cols] = 0.0
        printColumns(X)

        def _getFeatureStats(row, col, stats):
            featureStats = stats[FEATURES_KEY][col]
            if featureStats is None:
                logger.warning(f'column {col} not found in featureStats')
                return None, None
            col_n = featureStats['col']
            if col_n is None:
                logger.warning(f'col {col} not found in featureStats')
                return None, None
            return featureStats, col_n

        def _addMissingValueStats(featureStatsValues, value, col):
            if isinstance(value, (int, float)):
                if value == float(int(value)):
                    if value == 0:
                        tryValue1 = value + 1.0
                        tryValue2 = value + 2.0
                    else:
                        tryValue1 = value - 1.0
                        tryValue2 = value - 2.0
                    dictValue1 = featureStatsValues.get(tryValue1, None)
                    dictValue2 = featureStatsValues.get(tryValue2, None)
                    if dictValue1 is not None and dictValue2 is not None:
                        featureStatsValues[value] = {
                            'mean': dictValue1['mean'] - (dictValue2['mean'] - dictValue1['mean']),
                            '50%': dictValue1['50%'] - (dictValue2['50%'] - dictValue1['50%']),
                        }
                        logger.warning(
                            f'add missing value {value} for column {col} using {tryValue1} and {tryValue2}')
                        return
                elif (value - float(int(value))) == 0.5:
                    tryValue1 = value - 0.5
                    tryValue2 = value + 0.5
                    dictValue1 = featureStatsValues.get(tryValue1, None)
                    dictValue2 = featureStatsValues.get(tryValue2, None)
                    if dictValue1 is not None and dictValue2 is not None:
                        featureStatsValues[value] = {
                            'mean': (dictValue1['mean'] + dictValue2['mean']) / 2.0,
                            '50%': (dictValue1['50%'] + dictValue2['50%']) / 2.0,
                        }
                        logger.warning(
                            f'add missing value {value} for column {col} betwen {tryValue1} and {tryValue2}')
                        return
            featureStatsValues[value] = {'mean': 0, '50%': 0}
            logger.warning(
                f'add missing value {value} for column {col} as zero')

        def _transformDiscrete(row, col, stats):
            featureStats, col_n = _getFeatureStats(row, col, stats)
            if featureStats is None:
                return
            # set feature value
            if featureStats.get(VALUES_KEY, None) is None:
                logger.warning(
                    f"column {col} has no valueStats. {featureStats} {row['prov']} {row['city']} {row['ptype2-l']}")
                return
            valueStats = featureStats[VALUES_KEY].get(row[col_n], None)
            if valueStats is None:
                logger.warning(
                    f'column {col} value {row[col_n]} not found in valueStats. {featureStats}')
                _addMissingValueStats(
                    featureStats[VALUES_KEY], row[col_n], col)
                return
            if row[col+EXT_N] != 0:
                logger.warning(
                    f'column {col} has value {row[col+EXT_N]} already')
            row[col+EXT_N] = valueStats['mean']
            row[col+EXT_M] = valueStats['50%']

        def _transformContinuous(row, col, stats):
            featureStats, col_n = _getFeatureStats(row, col, stats)
            if featureStats is None:
                return
            # set feature value
            ratioStats = featureStats.get(RATIOS_KEY, None)
            if ratioStats is None:
                return
            row[col+EXT_N] = float(ratioStats['mean'] *
                                   row[col_n] + stats['mean'])
            row[col+EXT_M] = float(ratioStats['50%'] *
                                   row[col_n] + stats['mean'])

        def _transform(row):
            nonlocal totalCount, transformedCount
            totalCount += 1
            key = (row['prov'], row['city'], row['ptype2-l'])
            if key[2] == DROP:
                return row
            saletp = row[SALETP_COL]
            if saletp not in [0, 1]:
                logger.warning(f'invalid saletp {saletp}')
                return row
            saletp = int(saletp)
            stats = self.stats_[saletp].get(key, None)
            if stats is None:
                return row
            for col in self.discrete_cols:
                _transformDiscrete(row, col, stats)
            if saletp == 1:
                return row
            for col in self.continous_cols:
                _transformContinuous(row, col, stats)
            transformedCount += 1
            return row
        
        X = X.apply(_transform, axis=1)
        timer.stop(totalCount)
        logger.info(f'transformed {transformedCount}/{totalCount}')
        printColumns(X)
        return X
