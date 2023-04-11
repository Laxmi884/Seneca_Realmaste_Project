from base.const import MODEL_TYPE_REGRESSION, TRAINING_MIN_ROWS
from base.timer import Timer
from base.util import logDataframeChange
from data.estimate_scale import EstimateScale
from estimator.rmbase_estimate_manager import RmBaseEstimateManager
import lightgbm as lgb
import pandas as pd
from math import isnan
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class LgbmEstimateManager(RmBaseEstimateManager):
    """LightGBM manager."""

    model_name = 'lgbm'
    date_span = 365
    suffix_list = ['-n', '-c', '-b', '-e']
    numeric_columns_only = True
    default_model_params = {
        'n_estimators': 300,
        'max_depth': -1,
        'num_leaves': 100,
    }

    def __init__(
        self,
        data_source,
        name,
        model_params,
        estimate_both: bool = None,
        min_output_value: float = None,
        max_output_value: float = None,
    ):
        super().__init__(
            data_source,
            name,
            model_class=MODEL_TYPE_REGRESSION,
            estimate_both=estimate_both,
            min_output_value=min_output_value,
            max_output_value=max_output_value,
        )
        self.model_params = model_params

    def prepare_model(self):
        """Prepare model."""
        model_params = self.model_params or self.default_model_params
        self.model = lgb.LGBMRegressor(**model_params)
        self.logger.info('model_params: {model_params}')
        return self.model

    def filter_data(
        self,
        X,
    ):
        """ Filter data for LGBMRegressor """
        origX = X
        # remove columns with all NaN
        X = X.dropna(axis='columns', how='all')
        # remove columns with all zeros
        X = X.loc[:, (X != 0).any(axis=0)]
        # remove rows with NaN
        X = X.dropna()
        logDataframeChange(origX, X, self.logger, self.name)
        return X

    def train_single_scale(self, scale: EstimateScale) -> tuple[EstimateScale, object, float, list[str], dict]:
        # PCA should be added here
        timer = Timer(str(scale), self.logger)
        timer.start()
        df = self.my_load_data(scale)
        #df.head().to_excel("sh.xlsx")
        if df is None or df.shape[0] < TRAINING_MIN_ROWS:
            self.logger.info(
                '================================================')
            self.logger.warning(
                f'{str(scale)} {str(self.model_name)} No data for training')
            self.logger.info(
                '------------------------------------------------')
            return (None, None, None, None, None, None)
        
        
        model = self.prepare_model()
        x_cols, y_col, x_means = self.get_x_y_columns(df)
        df = self.filter_data_outranged(df, y_col=y_col) 
        
        
        y = StandardScaler()
        
        spare = pd.DataFrame(y.fit_transform(df[x_cols]), columns = x_cols)
        df = pd.concat([spare, pd.Series(df[y_col].tolist())], axis = 1)
        df = df.rename(columns={0: y_col})
        
        
        if y_col == "sp-n":
            col_names = pd.DataFrame(df.columns)
            col_names.to_excel("names.xlsx")
            
        df.to_excel("final_data3.xlsx")
        
        
        if df.shape[0] < TRAINING_MIN_ROWS:
            self.logger.info(
                '================================================')
            self.logger.warning(
                f'{str(scale)} {str(self.model_name)} No enough data for training after filter. {df.shape[0]} rows')
            self.logger.info(
                '------------------------------------------------')
            return (None, None, None, None, None, None)
        
        """
        pca = PCA(n_components=0.95) # preserve 95% of explained variance
        tt = df.iloc[:, 0:df.shape[1]-1]
        pd.DataFrame(tt.shape[0] - tt.count()).to_excel("null.xlsx")
        red_95 = pca.fit_transform(df.iloc[:, 0:df.shape[1]-1])
        df = pd.concat([pd.DataFrame(red_95),pd.DataFrame(df.iloc[:, df.shape[1]-1])], axis=1) 
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:df.shape[1]-1],df.iloc[:, df.shape[1]-1], test_size=0.15, random_state=10)
        model.fit(X_train, y_train)
        self.fit_output_min_max(df.iloc[:, df.shape[1]-1])
        accuracy = self.test_accuracy(model, X_test, y_test)
        #timer.stop(X_train.shape[0])
        if y_col == "sp-n":
            scores = cross_val_score(model, X_train, y_train, cv=10).mean()
            score_data = pd.DataFrame({f'Mean Validation Accuracy for {y_col}': [scores]})
            table = pd.read_excel('Cross_Val_Eval_1.xlsx') 
            table = pd.concat([table,score_data], axis = 0)
            with pd.ExcelWriter('Cross_Val_Eval_1.xlsx') as writer:
                table.to_excel(writer, index=False)
       """ 
        
        
        df_train, df_test = train_test_split(
            df, test_size=0.15) # random_state=10
        
        # cross only for price
        
        #if y_col == "sp-n":
        scores = cross_val_score(model, df_train[x_cols], df_train[y_col], cv=10).mean()
        score_data = pd.DataFrame({f'Mean Validation Accuracy for {y_col}': [scores]})
        table = pd.read_excel('Cross_Val_Eval_1.xlsx') 
        table = pd.concat([table,score_data], axis = 0)
        with pd.ExcelWriter('Cross_Val_Eval_1.xlsx') as writer:
            table.to_excel(writer, index=False)
        
        """
        scores = cross_val_score(model, df_train[x_cols], df_train[y_col], cv=10).mean()
        score_data = pd.DataFrame({f'Mean Validation Accuracy for {y_col}': [scores]})
        table = pd.read_excel('Cross_Val_Eval.xlsx') 
        table = pd.concat([table,score_data], axis = 0)
        with pd.ExcelWriter('Cross_Val_Eval.xlsx') as writer:
            table.to_excel(writer, index=False)

         """  
        
        model.fit(df_train[x_cols], df_train[y_col])
        self.fit_output_min_max(df[y_col])
        accuracy = self.test_accuracy(
            model, df_test[x_cols], df_test[y_col])
        
        

        accuracy_table = pd.read_excel('accuracy_no_changes.xlsx') #accuracy_no_changes #accuracy_with_changes
        new_data = pd.DataFrame({f'Accuracy for {y_col}:': [accuracy/100]})
        
        accuracy_table = pd.concat([accuracy_table,new_data], axis = 0)

        with pd.ExcelWriter('accuracy_no_changes.xlsx') as writer:
            accuracy_table.to_excel(writer, index=False)

        timer.stop(df_train.shape[0]) #timer.stop(df_train.shape[0])
        
        self.logger.info('================================================')
        self.logger.info(
            f'{str(scale)} {str(self.model_name)} model trained accuracy:{accuracy/100.0}%')
        featureImportance = self.feature_importance(model)
        self.logger.info('------------------------------------------------')
        featureImportanceDic = {}
        weightToPrint = []
        for weight, feature in featureImportance:
            intWeight = int(weight)
            featureImportanceDic[feature] = intWeight
            if intWeight == 0:
                weightToPrint.append(f'|{feature}')
            else:
                weightToPrint.append(
                    f'{feature.rjust(12)}:{str(weight).ljust(5)};')
        self.logger.info(''.join(weightToPrint))
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
