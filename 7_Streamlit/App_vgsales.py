import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFwe
from tpot.builtins import StackingEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVR
from sklearn.linear_model import LassoLarsCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
import numpy as np
import re

    
st.sidebar.title("Navigation")

pages=['Présentation du projet', 'Data Visualisation', 'Modélisation']
models= ["Regression Logistique", "KNN", "Random forest", 'Lasso', 'LinearSVR', 'LassoLarsCV']
page=st.sidebar.radio("Choisissez votre page", pages)


df_clean=pd.read_csv("df_clean_v2.csv", index_col=0)
df_gd=pd.get_dummies(df_clean)

y=df_gd['Global_Sales']
X=df_gd.drop('Global_Sales', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)


# Normaliser les variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pipeline3 = make_pipeline(
    StandardScaler(),
    MinMaxScaler(),
    SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
    VarianceThreshold(threshold=0.1)
)


def get_score(model):
    if model == models[0]:
        clf=LogisticRegression()
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    if model == models[1]:
        clf=KNeighborsRegressor()
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    if model == models[2]:
        clf=RandomForestRegressor()
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)
    
    if model == models[3]:
        clf=Lasso()
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    if model == models[4]:
        clf=LinearSVR()
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    if model == models[5]:
        clf=LassoLarsCV()
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)
    

   
        
if pages == pages[0]:
    st.title("Video Games Sales Project")
    st.header("This is a Machine Learning Project")
    st.subheader("by Débora Mandon")
    st.image('image-jeux-video.jpg')
    #st.markdown("This is my [dataframe](https://www.kaggle.com/competitions/titanic) :")
    st.dataframe(df_clean.head(10))


if pages == pages[1]:
    fig1=plt.figure()
    sns.countplot(x='Global_Sales', data = df_clean)
    st.pyplot(fig1)

       
if pages == pages[2]:
    model = st.selectbox("Choisissez votre modèle", models)
    st.write("Score obtenu", get_score(model))
    
