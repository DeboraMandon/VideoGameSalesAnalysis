import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFwe
from tpot.builtins import StackingEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVR
from sklearn.linear_model import LassoLarsCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import re
import joblib
import optuna
from joblib import dump, load


df_clean=pd.read_csv("df_clean_v2.csv", index_col=0)
df_gd=pd.get_dummies(df_clean)


y=df_gd['Global_Sales']
X=df_gd.drop('Global_Sales', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)


parameters_lr = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'positive' : [True, False],
    'copy_X': [True, False],
    'n_jobs': [-1, 10]
}

lr=LinearRegression()
grid_search = GridSearchCV(lr, parameters_lr, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params_lr = grid_search.best_params_
best_score_lr = -grid_search.best_score_ # Négation pour obtenir la valeur de MSE

lr_final = LinearRegression(**best_params_lr)
lr_final.fit(X_train, y_train)

y_pred_lr = lr_final.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
dump(best_params_lr, 'lr.joblib')        

parameters_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size' : range(1, 100),
    'metric' : ['minkowski', 'euclidean', 'manhattan', 'chebyshev']            
}

knn=KNeighborsRegressor()
grid_search = GridSearchCV(knn, parameters_knn, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params_knn = grid_search.best_params_
best_score_knn = -grid_search.best_score_ # Négation pour obtenir la valeur de MSE

knn_final = KNeighborsRegressor(**best_params_knn)
knn_final.fit(X_train, y_train)

y_pred_knn = knn_final.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
            

parameters_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 'auto', None],
    'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'min_weight_fraction_leaf' : range(0.1, 10.0)
    
}

rf=RandomForestRegressor()
grid_search = GridSearchCV(rf, parameters_rf, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params_rf = grid_search.best_params_
best_score_rf = -grid_search.best_score_ # Négation pour obtenir la valeur de MSE

rf_final = RandomForestRegressor(**best_params_knn)
rf_final.fit(X_train, y_train)

y_pred_rf = rf_final.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
        

parameters_lass = {
    'alpha': [0.1, 0.5, 1.0],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'max_iter': [1000, 2000, 3000],
    'precompute' : [True, False],
    'copy_X' : [True, False],
    'positive' : [True, False],
    'selection' : ['cyclic', 'random']
}

lass=Lasso()
grid_search = GridSearchCV(lass, parameters_lass, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params_lass = grid_search.best_params_
best_score_lass = -grid_search.best_score_ # Négation pour obtenir la valeur de MSE

lass_final = Lasso(**best_params_knn)
lass_final.fit(X_train, y_train)

y_pred_lass = lass_final.predict(X_test)
mse_lass = mean_squared_error(y_test, y_pred_lass)
        

parameters_line = {
    'epsilon': [0.1, 0.2, 0.5],
    'C': [0.1, 1, 10],
    'fit_intercept': [True, False],
    'max_iter': [1000, 2000, 3000],
    'loss' : ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'dual' : [True, False],
    'intercept_scaling' : range(1, 100)
}
 
line=LinearSVR()
grid_search = GridSearchCV(line, parameters_line, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params_line = grid_search.best_params_
best_score_line = -grid_search.best_score_ # Négation pour obtenir la valeur de MSE

line_final = LinearSVR(**best_params_line)
line_final.fit(X_train, y_train)

y_pred_line = line_final.predict(X_test)
mse_line = mean_squared_error(y_test, y_pred_line)
        


lasso_cv = LassoLarsCV(cv=5)
lasso_cv.fit(X_train, y_train)

best_alpha_lasso_cv = lasso_cv.alpha_
best_mse_lasso_cv = lasso_cv.mse_path_.mean(axis=1).min()

y_pred_lasso_cv = lasso_cv.predict(X_test)
mse_lasso_cv = mean_squared_error(y_test, y_pred_lasso_cv)


'''
Ridge Regression : Ridge
ElasticNet : ElasticNet
Support Vector Regression (SVR) : SVR
Gradient Boosting Regression : GradientBoostingRegressor
Decision Tree Regression : DecisionTreeRegressor
Bayesian Ridge Regression : BayesianRidge
Passive Aggressive Regression : PassiveAggressiveRegressor
Extra Trees Regression : ExtraTreesRegressor
AdaBoost Regression : AdaBoostRegressor
Gaussian Process Regression : GaussianProcessRegressor
XGBoost Regression : XGBRegressor
LightGBM Regression : LGBMRegressor
CatBoost Regression : CatBoostRegressor
'''  