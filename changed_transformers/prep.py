import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import gamma,lognorm,beta,expon,norm,iqr, scoreatpercentile
from scipy.optimize import minimize_scalar
from sklearn.svm import OneClassSVM
import random
import re
from scipy.stats import norm, lognorm, weibull_min, expon, gamma, uniform, kstest

one_hot_names = []


# outliers removal with iqr and z score are the most simple yet effective

class OutlierRemover_general_iqr(BaseEstimator,TransformerMixin): # our own class to remove outliers - we will insert it to the pipeline 
    def __init__(self,cols=[],factor=1.5):
        self.factor = factor # higher the factor, extreme would be the outliers removed.
        self.cols = cols
        
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
            if X.columns[i] in self.cols:
                continue
            x = X.iloc[:, i].copy() # change the copy
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = self.median[i] # replace outliers with the median
            X.iloc[:, i] = x # make the column copy

        return X # our transformed df
    
class Outlier_Remover_with_Z(BaseEstimator,TransformerMixin): # using Standard score, 
    def __init__(self,cols=[]): # the same results as with iqr
        self.stats = {}
        self.cols = cols
    
    def fit(self,X,y=None):
        for col in X.columns:
            if col in self.cols:
                continue
            mean = X[col].mean()
            std =  X[col].std()
            median = X[col].median()
            self.stats[col] = [mean,std,median]
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        formula = lambda x,mean,std: (x-mean)/(std) # convert to Z-score
        for col in X.columns:
            if col in self.cols:
                continue
            cop = X[col].copy()
            mean, std,median = self.stats[col]
            ser = X[col].apply(lambda v: formula(v,mean,std))
            bool_series = ((ser <= -3) & (ser >= 3)) # those who are more thant 3 std from mean
            cop[bool_series] = median # replace with med
            X[col] =  cop
        return X
    

    
class Outliers_removal_ml_dif_rand(BaseEstimator,TransformerMixin): # # tested, results are not  good 
    def __init__(self,cols=[]):
        self.data = {}
        self.scaler = StandardScaler()
        self.cols = cols
    
    def fit(self,X,y=None):
        length = len(X.columns)
        cur_perc = 0
        
        for col in X.columns:
            
            if col in self.cols:
                cur_perc += 1
                print(round(cur_perc/length, 2)*100)
                print("!")
                continue
            
            d = X[[col]]
            d = self.scaler.fit_transform(d)            
            model = OneClassSVM(kernel='rbf', gamma='auto')
            model.fit(d)
            outliers = model.predict(d) == -1 
            self.data[col] = outliers
            cur_perc += 1
            print(round(cur_perc/length, 2)*100)
            print()
        return self
    
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy().reset_index(drop=True) 
        for col in X.columns:
            
            if col in self.cols:
                continue
            
            x = X[col].copy()
            outlier_inds = self.data[col]  
            l = sum(outlier_inds)  
            h = X[col].tolist() # data for column
            f = Fitter(h, # try these 5 distrs
           distributions=['gamma', 
                          'lognorm',
                          "expon",
                          "norm",
                          "uniform"]) 
            f.fit()          
            try:
                d = f.get_best(method = 'sumsquare_error') 
                best_distr = list(d.keys())[0]
                params = d[best_distr] 
                l = sum(outlier_inds)        
                if best_distr == "gamma":
                    a_gamma, scale_gamma = params["a"],params["scale"]
                    x[outlier_inds] =  np.random.gamma(a_gamma, scale=scale_gamma, size=l) 
                elif best_distr == "lognorm":
                    s_lognorm, scale_lognorm = params["loc"],params["scale"]
                    x[outlier_inds] = np.random.lognormal(s_lognorm, scale_lognorm, l)       
                elif best_distr == "expon": 
                    scale = params["scale"]
                    x[outlier_inds] = np.random.exponential(scale, l)  
                elif best_distr == "norm": 
                    mu_norm, std_norm = params["loc"],params["scale"]
                    x[outlier_inds] = np.random.normal(mu_norm, std_norm, l)
                elif best_distr == "uniform": 
                    a,b = params["loc"],params["scale"]
                    x[outlier_inds] = np.random.uniform(a, b, l)
            except KeyError: 
                x[outlier_inds]= x.median()
            X[col] = x
        return X
    
class OutlierRemover_distrs(BaseEstimator,TransformerMixin): # tested, results are  good 
    """
    get more accurate iqr accounting for distr
    """
    def __init__(self,cols = []):
        self.fitters = {}
        self.cols = cols
    def cal_bounds(self,best_distr,data,params): # get the bunds for different distrs
        if best_distr == "gamma":
            shape, loc, scale = params["shape"],params["loc"],params["scale"]
            theoretical_quantiles = gamma.ppf(np.linspace(0.01, 0.99, 99), shape, loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = gamma.ppf(0.25 - 1.5*iqr_differences, shape, loc, scale)
            upper_bound = gamma.ppf(0.75 + 1.5*iqr_differences, shape, loc, scale)
            
        elif best_distr == "lognorm":
            s, loc, scale = params["s"],params["loc"],params["scale"]
            theoretical_quantiles = lognorm.ppf(np.linspace(0.01, 0.99, 99), s, loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = lognorm.ppf(0.25 - 1.5*iqr_differences, s, loc, scale)
            upper_bound = lognorm.ppf(0.75 + 1.5*iqr_differences, s, loc, scale)
            
        elif best_distr == "expon":
            loc, scale = params["loc"], params["scale"]
            theoretical_quantiles = expon.ppf(np.linspace(0.01, 0.99, 99), loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = expon.ppf(0.25 - 1.5*iqr_differences, loc, scale)
            upper_bound = expon.ppf(0.75 + 1.5*iqr_differences, loc, scale)
        
        elif best_distr == "uniform":
            a,b = params["loc"],params["scale"]
            theoretical_quantiles = uniform.ppf(np.linspace(0.01, 0.99, 99), a, b)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = uniform.ppf(0.25 - 1.5*iqr_differences, a, b)
            upper_bound = uniform.ppf(0.75 + 1.5*iqr_differences, a, b)       
        else: # norm
            loc, scale = params["shape"],params["loc"],params["scale"]
            theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, 99), loc, scale)
            observed_quantiles = np.percentile(data, np.linspace(1, 99, 99))
            differences = np.abs(theoretical_quantiles - observed_quantiles)
            iqr_differences = iqr(differences)
            q1 = scoreatpercentile(data, 25)
            q3 = scoreatpercentile(data, 75)
            lower_bound = norm.ppf(0.25 - 1.5*iqr_differences, loc, scale)
            upper_bound = norm.ppf(0.75 + 1.5*iqr_differences, loc, scale)       
        return lower_bound,upper_bound
        
    def fit(self,X,y=None):
        for col in X.columns:
            if col in self.cols:
                continue
            
            h = X[col].tolist()
            f = Fitter(h, # try these 5
           distributions=['gamma', # gamma
                          'lognorm', # lognormal
                          "expon", # exp
                          "norm",
                          "uniform"]) # gauss
            f.fit()
            self.fitters[col] = f         
        return self
    
    @staticmethod
    def neg_log_likelihood(lam, data): # max likelyhood method
        n = len(data)
        log_likelihood = n * np.log(lam) - lam * np.sum(data)
        return -log_likelihood

    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.cols:
                continue
            
            f = self.fitters[col]
            data = X[col].tolist()
            x = X[col].copy() 
            try:
                d = f.get_best(method = 'sumsquare_error')
                best_distr = list(d.keys())[0] # norm
                params = d[best_distr] # params of distr
                lower_bound,upper_bound = self.cal_bounds(best_distr,data,params) # get the upper and lower bound for each distr
                l = len(x[(x < lower_bound) | (x > upper_bound)])          
                if best_distr == "gamma":
                    a_gamma, scale_gamma = params["a"],params["scale"] #alpha, beta = params["shape"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = np.random.gamma(a_gamma, scale=scale_gamma, size=l) #[random.gammavariate(alpha, beta) for i in range(l)]
                elif best_distr == "lognorm":
                    mu, sigma = params["loc"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = np.random.lognormal(mu, sigma, l) #[random.lognormvariate(mu, sigma) for i in range(l)]                   
                elif best_distr == "expon":
                    res = minimize_scalar(self.neg_log_likelihood, args=(data,)) #OutlierRemover_distrs (use max likeluhhod method to estimate the lambda since mean could be 0 and lambda = 1/mean)
                    lambda_exp = res.x
                    x[(x < lower_bound) | (x > upper_bound)] =  np.random.exponential(lambda_exp, l)  #[random.expovariate(lambda_exp) for i in range(l)]            
                elif best_distr == "norm": 
                    mu, sigma = params["loc"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = np.random.normal(mu, sigma, l) #[random.gauss(mu, sigma) for i in range(l)]   
                elif best_distr == "uniform": 
                    a,b = params["loc"],params["scale"]
                    x[(x < lower_bound) | (x > upper_bound)] = np.random.uniform(a, b, l)
                else:
                    print("!"*100)
                    continue
            except KeyError:
                i = OutlierRemover_general_iqr()
                x = i.fit_transform(pd.DataFrame(x))

            X[col] = x 
        return X
    
"""  
class Outliers_removal_ml_dif_rand(BaseEstimator,TransformerMixin): # not working class
    def __init__(self):
        self.data = {}
        self.scaler = StandardScaler()
    
    def fit(self,X,y=None):
        length = len(X.columns)
        cur_perc = 0
        
        for col in X.columns:  # for each col
            d = X[[col]]
            d = self.scaler.fit_transform(d)
            model = OneClassSVM(kernel='rbf', gamma='auto')#OneClassSVM(kernel= "linear") # train binary SVM 
            model.fit(d)
            outliers = model.predict(d) == -1 # the outliers will be -1 else 1, so bool vector True is outlier else False
            self.data[col] = outliers
            cur_perc += 1
            print(round(cur_perc/length, 2)*100)
        return self
    
    def transform(self,X,y=None):
        print("!!!")
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            x = X[col].copy()
            data  = X[col]
            outlier_inds = self.data[col]  
            l = sum(outlier_inds)  
            
            mu_norm, std_norm = norm.fit(data)
            s_lognorm, loc_lognorm, scale_lognorm = lognorm.fit(data)
            c_weibull, loc_weibull, scale_weibull = weibull_min.fit(data)
            loc_expon, lambda_expon = expon.fit(data)
            a_gamma, loc_gamma, scale_gamma = gamma.fit(data)
            loc_unif, scale_unif = uniform.fit(data)

            ks_norm = kstest(data, norm(mu_norm, std_norm).cdf)[1]
            ks_lognorm = kstest(data, lognorm(s_lognorm, loc_lognorm, scale_lognorm).cdf)[1]
            ks_weibull = kstest(data, weibull_min(c_weibull, loc_weibull, scale_weibull).cdf)[1]
            ks_expon = kstest(data, expon(loc_expon, lambda_expon).cdf)[1]
            ks_gamma = kstest(data, gamma(a_gamma, loc_gamma, scale_gamma).cdf)[1]
            ks_unif = kstest(data, uniform(loc_unif, scale_unif).cdf)[1]

            p_values = [ks_norm, ks_lognorm, ks_weibull, ks_expon, ks_gamma, ks_unif]
            distr_names = ['normal', 'lognormal', 'weibull', 'exponential', 'gamma', 'uniform']
            best_fit = distr_names[np.argmin(p_values)]
            if best_fit == 'normal':
                rands = np.random.normal(mu_norm, std_norm, l)
            elif best_fit == "lognormal":
                rands =  np.random.lognormal(s_lognorm, scale_lognorm, l)
            elif best_fit == "weibull":
                rands = np.random.weibull(c_weibull, l) * scale_weibull
            elif best_fit == "exponential":
                rands = np.random.exponential(lambda_expon, l)
            elif best_fit == "gamma":
                rands = np.random.gamma(a_gamma, scale=scale_gamma, size=l)
            elif best_fit == "uniform":
                rands = np.random.uniform(loc_unif, scale_unif, l)
                
            x[outlier_inds] = rands 
            X[col] = x
        return X.reset_index(drop=True)
    
"""
        
class Custom_Cat_Imputer(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy() # make the copy
        values = {col:X[col].mode()[0] for col in X.columns} # replace with correposnding mode
        X = X.fillna(value=values)

        return X # our transformed df

# not using currently (converts to one hot encoding and replacing nulls with the mode)
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
        X = pd.DataFrame(X).copy()
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X),columns = X.columns) # impute nulls
        X_imputed = X_imputed.apply(self.get_rep_value) # get rid of the "numeric" objects in feature
        X_one_hot_encoded = self.one_hot_encoder.fit_transform(X_imputed) # encode
        self.column_names = self.one_hot_encoder.get_feature_names(X_imputed.columns) # the encoded names
        X_df = pd.DataFrame(X_one_hot_encoded.toarray(), columns=self.column_names)
        one_hot_names = self.column_names
        return X_df


class null_imputer_corr(BaseEstimator, TransformerMixin): # based on the linear correlation between features
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
    

class null_imputer_ml(BaseEstimator, TransformerMixin): # ml approach, currently used
    def __init__(self,dec = False):
        self.data = {}
        self.scaler = StandardScaler()
        self.dec = dec
    def fit(self,X,y=None):
        #X = pd.DataFrame(X).copy().reset_index(drop=True)
        #features = set(X.columns.tolist()) 
        m = X.isna() 
        length = m.shape[0]
        for col in X.columns:
            perc = sum(m[col])/length
            
            if round(perc,1) <= 0.2: # if perc of nulls in the folumn is less than 20%
                median = X[col].median() # we can just use the median to replace
                self.data[col] = median 
                
        return self
  
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy().reset_index(drop=True)    
        for col in list(self.data.keys()): # those with perc nulls less than 20%
            median = self.data[col]
            x = X[col].copy()
            x = x.fillna(median) # replace na with median
            X[col] = x      
        features_clean = list(self.data.keys())
        features_null = list(set(X.columns).difference(set(features_clean))) # our null features
        
        # scale first (clean features)
        sc = pd.DataFrame(self.scaler.fit_transform(X[features_clean]), columns=features_clean)
        
        for col in features_null: # then for each null feat
        
            if col in ["taxyr-n","rmBltYr-n","bltYr-n"]: # ignore the years # rmBltYr
                continue
            
            ser = X[col]
            test = list(ser[ser.isnull()].index)  # get lst of nulls inds
            train = list(ser[ser.notnull()].index)  # get lst of no nulls inds
            X_train = sc[features_clean].iloc[train, :] # use no null to train the feautures
            y_train = X[[col]].iloc[train, :]
            X_test = sc[features_clean].iloc[test, :] # null inds to predict after fit
            model = LinearRegression() if not self.dec else GradientBoostingRegressor() #RandomForestRegressor() 
            model.fit(X_train,y_train) 
            preds = model.predict(X_test)
            no_nulls = []
            if not self.dec: # dec tree gives an array while regression gives a vector (2 dim)
                for cube in preds: 
                    no_nulls.append(cube[0])
            else:
                no_nulls = preds            
            x = ser
            x[test] = no_nulls 
            #if col in ["taxyr","rmBltYr","bltYr"]:
                #x = x.apply(lambda z : round(z,0))
            X.loc[:, col] = x   
        return X


# not using currently

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
    
# not using currently

class Dates_numeric_Pipeline(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): 
        return self
    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].interpolate(method='linear').round(0)
            X[col] = X[col].interpolate(method='bfill', limit_direction = "backward")
        return X 
    
    
def deal_with_sqft(x): # used in fit method for preprocessor
    if x == None:
        return None
    if isinstance(x,int) or isinstance(x,float):
        return x
    try:
        if re.findall(r"\s*\d+\s*-\s*\d+\s*",x): # if num-num
            a,b = x.split("-")
            a = int(a.strip())
            b = int(b.strip()) 
            a = min([a,b])
            b = max([a,b])
            mean = (a+b)/2
            low = (a-b)/2 
            up = (b-a)/2
            return mean+ random.uniform(low, up) # get the mean plus some deviation
        elif re.findall(r"\s*\d+\s*\+\s*",x): # if num+
            num = float((x.strip()[:-1].strip()))
            return num + random.gauss(num, 20)
        elif re.findall(r"\s*[<]\s*\d+\s*",x): # if <num
            num = float(x.strip()[1:].strip())
            mean = num/2
            return mean+ random.uniform(0, mean-1)
        else: # just num
            return float(x)
    except TypeError: # if na 
        return None