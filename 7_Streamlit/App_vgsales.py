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



st.sidebar.title("Video Games Sales Analysis")
st.sidebar.image('gamer.png')
st.sidebar.subheader("Navigation")

pages=['Présentation du projet', 'Dataframe', 'Data Visualisation', 'Modélisation']
models= ["Regression Linéaire", "KNN", "Random forest", 'Lasso', 'LinearSVR', 'LassoLarsCV']
page=st.sidebar.radio("Choisissez votre page", pages)

        
if page == pages[0]:
    st.title("Video Games Sales Analysis")
    st.header("Machine Learning Project")
    st.subheader("by Débora Mandon")
    st.image('image-jeux-video.jpg')
    st.markdown("Pour ce projet il faudra estimer les ventes totales d’un jeu vidéo à l’aide d’informations descriptives comme: \n"
                "- Le pays d’origine \n"
                "- Le studio l’ayant développé ● \n"
                "- L’éditeur l’ayant publié \n"
                "- La description du jeu ● \n"
                "- La plateforme sur laquelle le jeu est sortie \n"
                "-  Le genre")
    st.markdown("")
    st.markdown("La difficulté du projet est que les données fournies ne contiennent que les ventes totales du jeu, le studio, le pays et l’éditeur. Le reste des données devra être scrappé sur des sites tels que Metacritic ou jeuxvideo.com. De plus, une étude d’analyse de sentiments pourra être effectuée afin de quantifier l’engouement généré par un jeu avant la sortie afin d’en prédire les ventes.")
    st.markdown("Attention, il y a énormément de scraping à faire, et il ne faudra pas avoir peur d’écrire des codes très techniques, de fouiller dans des documentations et d’apprendre à utiliser des librairies qui ne font pas partie de la formation.")
    st.markdown("")
    st.markdown("Données : Le dataset fourni est consultable [ici](https://www.kaggle.com/gregorut/videogamesales).")
    st.markdown("Les autres descripteurs ont été récupérés via du web scraping à l’aide de la libraire Selenium.")


if page == pages[1]:
    st.header("Analyse du Dataframe")
    st.image("analytics.jpg")
    st.markdown("")
    st.markdown("Notre dataset de base est composée de 16598 lignes et 10 colonnes. \n"
                "Il comprend les variables suivantes : \n"
                "- 0 Name - object(string) \n"
                "- 1 Platform - object(string) \n"
                "- 2 Year - float64 \n"
                "- 3 Genre - object (string) \n"
                "- 4 Publisher - object (string) \n"
                "- 5 NA_Sales - float64 \n"
                "- 6 EU_Sales - float64 \n"
                "- 7 JP_Sales - float64 \n"
                "- 8 Other_Sales - float64 \n"
                "- 9 Global_Sales - float64")            
    st.markdown("")
    st.markdown("Après le Webscrapping, nous avons obtenu les nouvelles variables suivantes : \n")   
    st.markdown("- developpeur_goo \n"
                "- PEGI_goo \n"
                "- developpeur_wiki \n"
                "- PEGI_wiki \n"
                "- Meta_NP \n"
                "- Meta_VP \n"
                "- Meta_NUsers \n"
                "- Meta_VUsers \n"
                "- meta_score \n"
                "- user_review \n"
                "- Rank \n"
                "- test_note_JVC \n"
                "- avis_count_JVC \n"
                "- avis_note_JVC \n"
                "- Classification_Age_JVC \n"
                "- max_joueurs_JVC \n"
                "- Support_JVC \n"
                "- langue_parlée_JVC \n"
                "- texte_JVC \n"
                "- nb_joueurs_JVC \n"
                "- game_mode \n"
                "- game_mode_JVC \n")
    st.markdown("")
    st.dataframe(df.head(10))
    st.markdown("")
    st.write("Shape du Dataframe :",df.shape)
    st.write("Nom des colonnes :",df.columns)
    st.write("Description du Dataframe :",df.describe())
    st.markdown("")
    st.markdown("Pour obtenir le jeu de données utilisé pour le Machine Learning, nous avons réalisé une étape \n"
                "de Pre Processing pour traiter les valeurs nulles, les variables très corrélées, supprimer les variables \n"
                "qui n'apportent aucune information... Nous avons également utilisé des méthode de Text Mining \n"
                "pour extraire les informations utiles des chaînes de caractères scrppés sur Internet.")
    st.markdown("")
    st.header("Mon Dataframe final")    
    st.write("Shape du nouveau Dataframe.v2 : ", df_clean.shape)
    st.write("Nom des colonnes :",df_clean.columns)
    st.write("Description du Dataframe.v2 :",df_clean.describe())    
    
if page == pages[2]:
    st.header("Data Visualisation") 
    st.subheader("Pour visualiser les données à l'aide de graphiques, choisissez un type de visualisation puis les variables à explorer."
                 )        
    graphs=["Evolution des ventes par Région", "Répartition des Ventes par Région", 
            "Répartition des 10 catégories les plus représentées par variables catégorielles", "Distribution des variables numériques",
            "Heatmap des variables numériques du Dataframe après PreProcessing"]
    
    graph=st.radio("Choisissez votre visualisation", graphs)
        
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

# Entrainement des modèles

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

def get_score(model):
    if model == models[0]:
        y=df_gd['Global_Sales']
        X=df_gd.drop('Global_Sales', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

        pipeline = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
            VarianceThreshold(threshold=0.1),
            LinearRegression())

        set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        y_pred_pipe = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[1]:
        num_n_neighbors=st.slider("Choisissez le nombre de voisins, n_neighbors : ", 1, 50, 17)
        num_weights=st.radio("Choisissez une fonction de pondération :", ['uniform', 'distance'])
        num_metrics=st.radio("Choisissez une fonction de distance entre les points :", ['minkowski', 'euclidean', 'manhattan', 'chebyshev'])
        num_algos=st.radio("Choisissez une fonction de distance entre les points :", ['auto', 'ball_tree', 'kd_tree', 'brute'])
        
        y=df_gd['Global_Sales']
        X=df_gd.drop('Global_Sales', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

        pipeline = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
            VarianceThreshold(threshold=0.1),
            KNeighborsRegressor(n_neighbors=num_n_neighbors, weights=num_weights, metric=num_metrics, algorithm=num_algos))
        pipeline.fit(X_train, y_train)
        y_pred_pipe = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[2]:
        y=df_gd['Global_Sales']
        X=df_gd.drop('Global_Sales', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

        pipeline = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
            VarianceThreshold(threshold=0.1),
            RandomForestRegressor())
        pipeline.fit(X_train, y_train)
        y_pred_pipe = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae
    
    if model == models[3]:
        num_alpha=st.slider("Choisissez la hauteur d'alpha :", 0.0, 10.0, 0.1)
        num_max_iter=st.slider("Choisissez le nombre maximum d'itérations effectuées :", 0, 10000, 10000)
        num_selection=st.radio("Choisissez la méthode utilisée pour sélectionner les variables dans le modèle :", ['cyclic', 'random', 'adaptive'])
        
        y=df_gd['Global_Sales']
        X=df_gd.drop('Global_Sales', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

        pipeline = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
            VarianceThreshold(threshold=0.1),
            Lasso(alpha=num_alpha, max_iter=num_max_iter, random_state=42, fit_intercept=True, selection=num_selection))
        pipeline.fit(X_train, y_train)
        y_pred_pipe = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[4]:
        num_max_iter=st.slider("Choisissez le nombre maximum d'itérations effectuées :", 0, 10000, 10000)
        num_C=st.slider("Choisissez la force de régularisation, C :", 0.1, 1000.0, 1.0)
        num_epsilon=st.slider("Choisissez la marge de tolérance de l'erreur, epsilon :", 0.1, 1000000.0, 0.1)
        num_loss=st.radio("Choisissez la fonction de perte utilisée pour optimiser le modèle :", ['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber'])
        num_verbose=st.slider("Choisissez le nombre d'information à afficher :", 0, 10, 1)
        
        y=df_gd['Global_Sales']
        X=df_gd.drop('Global_Sales', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

        pipeline = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
            VarianceThreshold(threshold=0.1),
            LinearSVR(C=num_C, epsilon=num_epsilon, loss=num_loss, max_iter=num_max_iter, verbose=num_verbose ))
        pipeline.fit(X_train, y_train)
        y_pred_pipe = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae

    if model == models[5]:
        num_alpha=st.slider("Choisissez la hauteur d'alpha :", 1, 10000, 1000)
        #num_verbose=st.slider("Choisissez le nombre d'information à afficher :", 0, 10, 1)
        
        y=df_gd['Global_Sales']
        X=df_gd.drop('Global_Sales', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

        pipeline = make_pipeline(
            StandardScaler(),
            MinMaxScaler(),
            SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
            StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
            VarianceThreshold(threshold=0.1),
            LassoLarsCV(max_n_alphas=num_alpha, verbose=False))
        pipeline.fit(X_train, y_train)
        y_pred_pipe = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred_pipe)
        mse = mean_squared_error(y_test, y_pred_pipe)
        mae = mean_absolute_error(y_test, y_pred_pipe)
        return "R2", r2, "MSE", mse, "MAE", mae
         
if page == pages[3]:
    st.header("Entrainement des modèles")
    model = st.radio("Choisissez votre modèle", models)
    st.write("Score obtenu", get_score(model))
    
    
    
