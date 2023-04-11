# Predictor

## Overview

Base knowledge:
(Grid Search)[https://scikit-learn.org/stable/modules/grid_search.html#grid-search]
(GridSearchCV)[https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV]
(Feature Selection)[https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection]

(Feature Extraction)[https://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction]
(Feature Scaling)[https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing]
(Feature Transformation)[https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing]

(Ensemble)[https://scikit-learn.org/stable/modules/ensemble.html#ensemble]

# Classes

- RmBaseEstimateManager
  is the base class for all estimator managers, each estimator manager is responsible for one type of estimation.
  - writeBack: write back to datasource
  - train: train the model
  - estimate: estimate the value
  - save: save the model
  - load: load the model
- RmBaseEstimator
  is the base class for all estimators, each estimator is responsible for one type of estimation for one scale.

  -

- Estimator: base class for all estimaters
  - public properties:
    - col_list: list of column names
    - model
    - model_params
    - data_range
    - X_train
    - y_train
    - X_test
    - y_test
    - X: input for prediction
  - public methods:
    - set_scale(EstimateScale):
    - train:
    - tune:
      - try different parameters
      - Dimensionality Reduction: PCA ...
    - predict(X)
    - load_data: mongodb or cvs
    - save_model: pickle
    - load_model:
    - save_params: json
    - load_params:
    - prepare_data(X_data: X_train/X_test/X)
- Subclasses of Estimator
  - Sqft: sqft and range
  - BuiltYear: built-year and range
  - Value: property value
  - Salability: percentage of salability on the given asking price
  - Dom: day on market if salable
  - Price: sold price on the given price
  - RentValue: rent value
  - PriceChange: price change on a month and two months
- EstimateScale: model data scale
  - TimePoint: 20210101
  - PropTypes: Detached, Semi-Detached, Townhouse, Condo
  - Prov: ON
  - Area: optional
  - City: optional
  - Cmty: optional
- DataStore: training data driver. connect to db

  - load_data: model_scale, timespan, fields

- Model:
  - properties:
    - name:
  - methods:
    - preprocess:
    - train:
    - predict:
    - test: return accuracy scores for the test data
    - tune: try different parameters for the model
    - save_model: save the model to folder
    - load_model: load the model from folder or file
- ML Models:
  - https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning#gs.fZ2A1Jk
  - LightGBM/Ensemble
  - RandomForest/Ensemble
  - XGboost/Ensemble
  - ElasticNet/Regularization:
  - Support Vector Machines (SVM)
  - kNN/K-Nearest Neighbors/Instance based:
  - Linear Regression/Regression
  - CART: Classification and regression tree/Decision Tree
  - k-Means/k-Medians/Clustering
  - Stacking/Blending(BaseModels,DecisionModle)
