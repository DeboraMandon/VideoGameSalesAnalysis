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
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import re

df= pd.read_csv("gaming_total_v2.csv")
df= df.drop('Unnamed: 0', axis=1)
df_clean=pd.read_csv("df_clean_v2.csv", index_col=0)
df_gd=pd.get_dummies(df_clean)

y=df_gd['Global_Sales']
X=df_gd.drop('Global_Sales', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

pipeline = make_pipeline(
    StandardScaler(),
    MinMaxScaler(),
    SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
    VarianceThreshold(threshold=0.1))

set_param_recursive(pipeline.steps, 'random_state', 42)
pipeline.fit(X_train, y_train)

st.sidebar.title("Navigation")

pages=['Présentation du projet', 'Data Visualisation', 'Modélisation']
models= ["Regression Linéaire", "KNN", "Random forest", 'Lasso', 'LinearSVR', 'LassoLarsCV']
page=st.sidebar.radio("Choisissez votre page", pages)

        
if page == pages[0]:
    st.title("Video Games Sales Project")
    st.header("This is a Machine Learning Project")
    st.subheader("by Débora Mandon")
    st.image('image-jeux-video.jpg')
    st.markdown("For this project it will be necessary to estimate the total sales of a video game using descriptive information.")
    st.markdown("This is my [Dataframe](https://raw.githubusercontent.com/DeboraMandon/video_game_sales_analysis/main/5_Stats_DF/gaming_total_v2.csv) :")
    st.dataframe(df.head(10))
    st.write("Shape du Dataframe :",df.shape)
    st.write("Nom des colonnes :",df.columns)
    st.write("Description du Dataframe :",df.describe())


if page == pages[1]:
    
    graphs=["Evolution des ventes par Région", "Répartition des Ventes par Région", 
            "Répartition des 10 catégories les plus représentées par variables catégorielles", "Distribution des variables numériques",
            "Heatmap du Dataframe"]
    
    graph=st.selectbox("Choisissez votre visualisation", graphs)
        
    if graph == graphs[0]:    
        for i in df.select_dtypes(include=['int64', 'float64']):
            df[f'cat_{i}'] = pd.qcut(df[i], q=[0,.25,.5,.75,1.], duplicates='drop')
        df['Year']=pd.to_datetime(df['Year'], format='%Y')
        sales_per_year=df.groupby('Year', as_index=False).agg({'NA_Sales':sum, 'EU_Sales':sum, 'JP_Sales':sum, 'Other_Sales':sum,'Global_Sales':sum})
        choices_num =st.multiselect("Choisissez les variables numériques à étudier", sales_per_year.select_dtypes(include=['int64', 'float64']).columns)
        
        fig1=plt.figure(figsize=(10,5))
        sns.set(style="whitegrid")
        plt.plot_date(x=sales_per_year['Year'].values,
                    y=sales_per_year[choices_num].values,
                    xdate=True,
                    ls='-',
                    label=choices_num)
        plt.legend(loc='best')
        plt.xlabel('Année')
        plt.ylabel('Ventes en millions')
        plt.title('Evolution des ventes par Région')
        st.pyplot(fig1)
    
    if graph == graphs[1]:
        fig2=plt.figure(figsize=(10,6))
        valeurs=df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
        plt.title('Répartition des Ventes par Région')
        plt.pie(x=valeurs.values,labels=valeurs.index,autopct=lambda x: f'{str(round(x, 2))}%');
        st.pyplot(fig2)
    
    if graph == graphs[2]:
        choices_cat =st.radio("Choisissez les variables catégorielles à étudier", df_clean.select_dtypes(include=['object']).columns)
        fig3=plt.figure()
        df.select_dtypes(include=['object'])[choices_cat].value_counts()[:10].plot.pie(autopct=lambda x: f'{str(round(x, 2))}%')
        plt.title('Répartition de la variable ' + str(choices_cat))
        st.pyplot(fig3)

    if graph == graphs[3]:
        choices_num =st.radio("Choisissez les variables numériques à étudier", df_clean.select_dtypes(include=['int64', 'float64']).columns[:-1])
        fig4=plt.figure()
        sns.distplot(df[choices_num], label=choices_num)
        plt.title('Histogramme de la variable ' + str(choices_num))
        plt.legend(loc='best')
        st.pyplot(fig4)
        
    if graph == graphs[4]:    
        fig5=plt.figure()
        sns.heatmap(df_clean.select_dtypes(include=['int64', 'float64']).corr(),annot=False)   
        st.pyplot(fig5)


def get_score(model):
    if model == models[0]:
        clf=LinearRegression()
        clf.fit(X_train, y_train)
        y_pred_pipe = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[1]:
        clf=KNeighborsRegressor()
        clf.fit(X_train, y_train)
        y_pred_pipe = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[2]:
        clf=RandomForestRegressor()
        clf.fit(X_train, y_train)
        y_pred_pipe = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae
    
    if model == models[3]:
        num_alpha=st.slider("Choisissez la hauteur d'alpha", 0.0, 10.0, 0.1)
        clf=Lasso(alpha=num_alpha, max_iter=10000, random_state=42, fit_intercept=True)
        clf.fit(X_train, y_train)
        y_pred_pipe = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[4]:
        clf=LinearSVR()
        clf.fit(X_train, y_train)
        y_pred_pipe = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[5]:
        clf=LassoLarsCV()
        clf.fit(X_train, y_train)
        y_pred_pipe = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae
         
if page == pages[2]:
    model = st.radio("Choisissez votre modèle", models)
    
    
    st.write("Score obtenu", get_score(model))
    
    
