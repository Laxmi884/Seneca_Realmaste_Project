# feature selection
from base.base_cfg import BaseCfg
from sklearn.feature_selection import SelectKBest, f_regression
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

logger = BaseCfg.getLogger(__name__)

# feature selection


def select_features(X_train, y_train, X_test=None):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    if X_test is not None:
        X_test_fs = fs.transform(X_test)
    else:
        X_test_fs = None
    return X_train_fs, fs, X_test_fs


def extract_raw_name(name: str) -> str:
    return name.split('-')[0]


def raw_feature_sort(nameScores: list[tuple]) -> list[tuple]:
    """Sort the features by their raw score.
    """
    nameScoreDict = {}  # name: score
    for name, score in nameScores:
        name = extract_raw_name(name)
        if name not in nameScoreDict:
            nameScoreDict[name] = score
        else:
            nameScoreDict[name] += score
    nameScoreList = sorted(
        nameScoreDict.items(),
        key=lambda x: x[1],
        reverse=True)
    return nameScoreList


def changeRate(scores):
    change = []
    for i in range(0, len(scores)-1):
        num = scores[i] - scores[i+1]
        change.append(num)
    return change


def plot_scores(self, scores: list[float]):
    # plot the scores
    # scores.sort()
    scores = scores.copy()
    scores.reverse()
    scores = np.flip(scores, axis=0)
    pyplot.bar([i for i in range(len(scores))], scores)
    pyplot.show()


class FeatureSelector():
    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str = None,
        prefer_estimated_value: bool = True,
    ):
        self.df = df
        self.y_col = y_col
        self.prefer_estimated_value = prefer_estimated_value
        self.logger = logger

    def prepare_data(
        self,
        df: pd.DataFrame = None,
        copy: bool = True,
        drop_any_na: bool = False,
    ):
        """Prepare data for training"""
        if df is None:
            df = self.df
        if copy:
            df = df.copy()
        rowsBefore = df.shape[0]
        colsBefore = df.shape[1]
        df.dropna(axis='columns', how='all', inplace=True)
        colsAfter = df.shape[1]
        if drop_any_na:
            df.dropna(inplace=True)
        else:
            df.dropna(axis='rows', how='all', inplace=True)
        rowsAfter = len(df.index)
        self.logger.info(
            f'*FeatureSelector* Rows dropped: {rowsBefore-rowsAfter}/{rowsBefore}=>{rowsAfter}\
 Cols dropped: {colsBefore-colsAfter}/{colsBefore}=>{colsAfter}')
        return df

    def generate_numeric_columns(self, df: pd.DataFrame = None):
        """Return the number of columns"""
        useSelf = False
        if df is None:
            df = self.df
            useSelf = True
        numeric_columns_ = []
        for col in df.columns:
            if '-' not in col:
                continue
            try:
                if (df[col].dtype == 'float64' or
                        df[col].dtype == 'int64'):
                    numeric_columns_.append(col)
                else:
                    self.logger.info(
                        f'{col} is not numeric got {df[col].dtype}')
            except Exception as e:
                self.logger.error(
                    f'Error in generating numeric column:{col} error:{e}')
        if self.prefer_estimated_value:
            new_cols = numeric_columns_.copy()
            for col in numeric_columns_:
                if col.endswith('-e'):
                    orig_col = col[:-2] + '_n'
                    if orig_col in numeric_columns_:
                        new_cols.remove(orig_col)
            numeric_columns_ = new_cols
        if useSelf:
            self.numeric_columns_ = numeric_columns_
        return numeric_columns_

    def fix_column_data(
        self,
        df: pd.DataFrame,
        cols: list[str] = None,
        keep_cols: list[str] = None,
        exclude_cols: list[str] = None,
    ):
        """Fix column data"""
        rows = df.shape[0]
        result_cols = cols.copy()
        keep_cols = keep_cols or []
        exclude_cols = exclude_cols or []
        for col in (cols or df.columns):
            na_rows = df[col].isna().sum()
            if na_rows > 0:
                self.logger.info(f'Fixing {col} {na_rows}/{rows}')
                if col in keep_cols:
                    self.logger.info(f'Keeping {col}')
                    continue
                elif (na_rows / rows < 0.5):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    self.logger.info(f'{col} is too many NAs. Droping it')
                    df.drop(col, axis=1, inplace=True)
                    result_cols.remove(col)
            if col in exclude_cols:
                result_cols.remove(col)
        return df, result_cols

    def find_features(self, y_col: str = None, exclude_cols: list[str] = None):
        if y_col is None:
            y_col = self.y_col
        if y_col is None:
            raise ValueError('y_col is None')
        df = self.prepare_data()
        cols = self.generate_numeric_columns(df)
        df = df[cols]
        df, cols = self.fix_column_data(
            df, cols, keep_cols=[y_col], exclude_cols=exclude_cols)
        df = self.prepare_data(df, copy=False, drop_any_na=True)
        x_cols = list(cols).copy()
        if y_col in x_cols:
            x_cols.remove(y_col)
        else:
            raise Exception(f'{y_col} not in {x_cols}')
        X_train = df[x_cols]
        y_train = df[y_col]
        X_selected, fs, _ = select_features(X_train, y_train)
        print(len(x_cols))
        scores = np.nan_to_num(fs.scores_, copy=True, nan=0)
        name_scores = zip(x_cols, scores)
        name_scores = list(name_scores)
        name_scores.sort(key=lambda tup: tup[1])
        name_scores.reverse()
        for k, v in name_scores:
            print(f'{k}:{v}')
        return name_scores, scores

    # return three levels of features and all features with scores
    def find_raw_features(self, y_col: str = None, exclude_cols: list[str] = None):
        name_scores, scores = self.find_features(y_col, exclude_cols)
        name_scores = raw_feature_sort(name_scores)
        scores = [x[1] for x in name_scores]
        changes = changeRate(scores)
        changeOfChanges = changeRate(changes)
        # lowest change index
        lowestChangeIndex = np.argmin(changeOfChanges)
        firstLevelFeatures = name_scores[0:lowestChangeIndex+2+1]
        remainingChangeOfChanges = changeOfChanges[lowestChangeIndex+1:]
        # second lowest change index
        secondLowestChangeIndex = np.argmin(remainingChangeOfChanges)
        secondLevelFeatures = name_scores[
            0:lowestChangeIndex+2+1+secondLowestChangeIndex+1]
        remainingChangeOfChanges = changeOfChanges[secondLowestChangeIndex+2:]
        # third lowest change index
        thirdLowestChangeIndex = np.argmin(remainingChangeOfChanges)
        thirdLevelFeatures = name_scores[
            0:lowestChangeIndex+2+1+secondLowestChangeIndex+1+thirdLowestChangeIndex+2]
        return [firstLevelFeatures, secondLevelFeatures, thirdLevelFeatures, name_scores]

# find_features_for_y('bltYr_n')
