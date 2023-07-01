# importer les librairies

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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LassoLarsCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from tpot.export_utils import set_param_recursive
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import re
import joblib
from joblib import dump, load
#import scikitplot as skplt
import shap
#from IPython.display import display
import configparser
import getpass

# télécharger les données puis les conserver en cache
@st.cache_data
def load_data():
    df= pd.read_csv("gaming_total_v2.csv")
    df= df.drop('Unnamed: 0', axis=1)
    df_clean=pd.read_csv("df_clean_v2.csv", index_col=0)
    df_gd=pd.get_dummies(df_clean)
    return df, df_clean, df_gd
df, df_clean, df_gd = load_data()

# créer le dataframe avec des nouvelles données pour tester le modèle
data_n = {
    "Platform": ["Xbox Series","PC", "PS5", "ONE", "Switch"],
    "Year": [2023, 2021, 2021, 2021, 2020],
    "Genre": ["Action", "Survival-Horror", "Action", "Action", "Aventure"],
    "Publisher": ["Warner Bros Games", "Capcom", "Sony Interactive Entertainment", "IO Interactive", "Nintendo"],
    "Meta_NP": [14, 24, 127, 33, 111],
    "Meta_VP": [43, 83, 88, 87, 90],
    "Meta_NUsers": [130, 1284, 2491, 234, 6594],
    "Meta_VUsers": [2.3, 6.9, 8.5, 8.3, 5.6],
    "test_note_JVC": [18, 16, 18, 17, 17],
    "avis_count_JVC": [61, 39, 219, 16, 425],
    "avis_note_JVC": [15.3, 16.2, 16.9, 16.4, 14],
    "Classification_Age_JVC": [8, 18, 8, 18, 3],
    "Support_JVC": ["DVD", "DVD", "Blu-ray", "Blu-ray", "cartouche"],
    "langue_parlée_JVC": ["français", "anglais", "français", "anglais", "français"],
    "texte_JVC": ["français", "français", "français", "français", "français"]}
df_new_data = pd.DataFrame(data_n)

# utilisateurs
username = getpass.getuser()

# liste des onglets
pages=['📖 Présentation du projet', '🗃️ Dataframe', '📈 Data Visualisation', '📊 PowerBI Rapport', 
    '🛠️ Hyperparamètres', '🚀 Modélisation', '💡 Interprétabilité des modèles', '🪄 Test du modèle']
# liste des modèles de ML
models= ["Regression Linéaire", "KNN", "Random forest", 'Lasso', 
        'LinearSVR', 'LassoLarsCV', 'SVR', 'DecisionTreeRegressor', 'AdaBoostRegressor']
# liste des graphiques de visualisation des modèles
graphes=["Graphique de régression", "Cumulative Gains Curve"]
# liste des graphiques d'étude du dataframe
graphs=["Evolution des ventes par Région", "Répartition des Ventes par Région", 
            "Répartition des catégories par variables catégorielles", #"Distribution des variables numériques",
            "Distribution des variables", "Heatmap des variables numériques du Dataframe après PreProcessing"]


## création de l'application sur streamlit

# création de la sidebar avec le logo de DataScientest
st.sidebar.image('DS.jpg')


# Authentification par mot de passe
st.sidebar.subheader("Authentification :")

def authentication():
    config = configparser.ConfigParser()
    config.read('cred.ini')
    mdp = config.get('Credentials', 'mdp')

    password = st.sidebar.text_input("Mot de passe :", type="password")
    if password == mdp:
        return True
    else:
        return False 

def main():
    if authentication():
        st.sidebar.markdown("<p style='color:red'>Application sécurisée</p>", unsafe_allow_html=True)
        st.write("")    
        st.sidebar.title("Video Games Sales Analysis")
        page=st.sidebar.radio("Choisissez votre page",pages)


        # Création du jeu d'entrainement et du jeu de test
        y=df_gd['Global_Sales']
        X=df_gd.drop('Global_Sales', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

        # fonction pour tester les paramètres des modèles
        def get_param(model):
            if model == models[0]:
                # récupère les meilleurs paramètres et les scores qui ont été enregistré sur le notebook jupyter param_joblib_pipe
                best_params_lr = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lr2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_lr2.joblib')
                best_model=LinearRegression(**best_params_lr)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de LinearRegressor : ")
                st.write(best_params_lr)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)
                return("Paramètres:", best_params_lr, "Score:", score, "MSE", mse)

            if model == models[1]:
                best_params_knn = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_knn2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_knn2.joblib')
                best_model=KNeighborsRegressor(**best_params_knn)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de KNeighborsRegressor : ")
                st.write(best_params_knn)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)
                return("Paramètres:", best_params_knn, "Score:", score, "MSE", mse)
                    
            if model == models[2]:
                best_params_rf = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_rf2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_rf2.joblib')
                best_model=RandomForestRegressor(**best_params_rf)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de RandomForestRegressor : ")
                st.write(best_params_rf)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)   
                return("Paramètres:", best_params_rf, "Score:", score, "MSE", mse)
            
            if model == models[3]:
                best_params_lass = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lass2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_lass2.joblib')
                best_model=Lasso(**best_params_lass)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de Lasso : ")
                st.write(best_params_lass)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)   
                return("Paramètres:", best_params_lass, "Score:", score, "MSE", mse)
            
            if model == models[4]:
                best_params_line = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_line2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_line2.joblib')
                best_model=LinearSVR(**best_params_line)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de LinearSVR : ")
                st.write(best_params_line)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)   
                return("Paramètres:", best_params_line, "Score:", score, "MSE", mse)
            
            if model == models[5]:
                best_params_lasso_cv = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lasso_cv2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_lasso_cv2.joblib')
                best_model=LassoLarsCV(**best_params_lasso_cv)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de LassoLarsCV : ")
                st.write(best_params_lasso_cv)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)  
                return("Paramètres:", best_params_lasso_cv, "Score:", score, "MSE", mse)

            if model == models[6]:
                best_params_svr = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_svr2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_svr2.joblib')
                best_model=SVR(**best_params_svr)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de SVR : ")
                st.write(best_params_svr)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)
                return("Paramètres:", best_params_svr, "Score:", score, "MSE", mse)
            
            if model == models[7]:
                best_params_dt = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_dt2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_dt2.joblib')
                best_model=DecisionTreeRegressor(**best_params_dt)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de DecisionTreeRegressor : ")
                st.write(best_params_dt)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)
                return("Paramètres:", best_params_dt, "Score:", score, "MSE", mse)
            
            if model == models[8]:
                best_params_ab = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_ab2.joblib') 
                best_score = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_score_ab2.joblib')
                best_model=AdaBoostRegressor(**best_params_ab)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                score=best_model.score(X_test, y_test)
                st.markdown("Meilleurs paramètres de AdaBoostRegressor : ")
                st.write(best_params_ab)
                st.markdown("Score : ")
                st.write(score)
                st.markdown("MSE : ")
                st.write(mse)
                return("Paramètres:", best_params_ab, "Score:", score, "MSE", mse)  

        # fonction pour obtenir les scores des modèles                
        def get_score(model):
            score_p = pipeline.score(X_test, y_test)
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            return(score_p, y_pred, r2, mse, mae)
        
        # fonction pour créer un graphique d'évaluation des modèles  
        def plot_perf(graphe_perf):
            if graphe_perf == graphes[0]:
                fig1=plt.figure()
                plt.scatter(y_test, y_pred, color='darkblue')
                plt.xlabel('Vraies valeurs')
                plt.ylabel('Prédictions')
                plt.title('Graphique de régression')
                st.pyplot(fig1)


        # PAGE 1 : page d'accueil, présentation du projet

        if page == pages[0]:
            st.image('image-jeux-video.jpg', use_column_width=1)
            st.title("Video Games Sales Analysis")
            st.header("Machine Learning Project")
            st.subheader("Auteurs : Débora Mandon, Guillaume Besançon, Severine Huang")
            st.markdown("Pour ce projet il faudra estimer les ventes totales d’un jeu vidéo à l’aide d’informations descriptives comme: \n"
                        "- Le pays d’origine \n"
                        "- Le studio l’ayant développé  \n"
                        "- L’éditeur l’ayant publié \n"
                        "- La description du jeu  \n"
                        "- La plateforme sur laquelle le jeu est sortie \n"
                        "-  Le genre")
            st.markdown("")
            st.markdown("La difficulté du projet est que les données fournies ne contiennent que les ventes totales du jeu, le studio, le pays et l’éditeur. Le reste des données devra être scrappé sur des sites tels que Metacritic ou jeuxvideo.com. De plus, une étude d’analyse de sentiments pourra être effectuée afin de quantifier l’engouement généré par un jeu avant la sortie afin d’en prédire les ventes.")
            st.markdown("Attention, il y a énormément de scraping à faire, et il ne faudra pas avoir peur d’écrire des codes très techniques, de fouiller dans des documentations et d’apprendre à utiliser des librairies qui ne font pas partie de la formation.")
            st.markdown("")
            st.markdown("Données : Le dataset fourni est consultable [ici](https://www.kaggle.com/gregorut/videogamesales).")
            st.markdown("Les autres descripteurs ont été récupérés via du web scraping à l’aide de la libraire Selenium.")

        # PAGE 2 : présentation des données, explication du jeu de donnée

        if page == pages[1]:
            st.sidebar.subheader("Dataframe")
            if st.sidebar.checkbox("Afficher les données brutes :", False):
                st.subheader("Jeu de données 'vg_sales' : Echantillon de 100 observations")
                st.write(df.sample(100))
            
            st.header("Analyse du Dataframe")
            st.markdown("")
            st.markdown("Notre dataset initial était composée de 16598 lignes et 10 colonnes. \n"
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
            st.write("Nom des colonnes :",df.columns.to_list())

            st.markdown("")
            st.markdown("")
            st.write("Shape du Dataframe :",df.shape)
            st.write("Description du Dataframe :",df.describe())
            st.markdown("")
            st.markdown("Pour obtenir le jeu de données utilisé pour le Machine Learning, nous avons réalisé une étape \n"
                        "de Pre Processing pour traiter les valeurs nulles, les variables très corrélées, supprimer les variables \n"
                        "qui n'apportent aucune information... Nous avons également utilisé des méthode de Text Mining \n"
                        "pour extraire les informations utiles des chaînes de caractères scrppés sur Internet.")
            st.markdown("")
            st.header("Le Dataframe utilisé pour l'entraînement du modèle")    
            st.write("Shape du nouveau Dataframe : ", df_clean.shape)
            st.write("Nom des colonnes :",df_clean.columns.to_list())
            st.write("Description du Dataframe :",df_clean.describe(), df_clean.describe(include='object'))  

        # PAGE 3 : data visualisation à l'aide de différents graphes

        if page == pages[2]:
            st.header("Data Visualisation")
            st.sidebar.header("Data Visualisation")
            graph=st.sidebar.selectbox("Choisissez votre visualisation", graphs)

            if graph == graphs[0]:    
                for i in df.select_dtypes(include=['int64', 'float64']):
                    df[f'cat_{i}'] = pd.qcut(df[i], q=[0,.25,.5,.75,1.], duplicates='drop')
                df['Year']=pd.to_datetime(df['Year'], format='%Y')
                sales_per_year=df.groupby('Year', as_index=False).agg({'NA_Sales':sum, 'EU_Sales':sum, 
                                                                    'JP_Sales':sum, 'Other_Sales':sum,'Global_Sales':sum})
                num_col_df=list(sales_per_year.select_dtypes(include=['int64', 'float64']).columns)
                choices_num =st.sidebar.multiselect("Choisissez les variables numériques à étudier", num_col_df, default=num_col_df)

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
                choices_cat =st.sidebar.radio("Choisissez les variables catégorielles à étudier", 
                                            df_clean.select_dtypes(include=['object']).columns)
                fig3=plt.figure()
                df_clean.select_dtypes(include=['object'])[choices_cat].value_counts().head(15).plot.pie(autopct=lambda x: f'{str(round(x, 2))}%')
                plt.title(f'Répartition de la variable {str(choices_cat)}')
                st.pyplot(fig3)

            if graph == graphs[3]:     
                choices_var =st.sidebar.radio("Choisissez les variables à étudier", df_clean.columns)
                fig5=plt.figure()
                sns.histplot(df_clean[choices_var], label=choices_var)
                plt.xticks(rotation=70, ha='right', fontsize=8)
                plt.legend()
                st.pyplot(fig5)
                    
            if graph == graphs[4]:    
                fig6=plt.figure()
                sns.heatmap(df_clean.select_dtypes(include=['int64', 'float64']).corr(),annot=False)   
                st.pyplot(fig6)

        # PAGE 4 : présentation PowerBI

        if page == pages[3]:
            st.header("Rapport PowerBI")    
            iframe_html = """ 
            <iframe title="Video_Games_Sales_Analysis - Analyse des ventes" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=6336e870-43dd-476a-b285-a4456b285cac&autoAuth=true&ctid=48f2d645-8d8d-4cf0-80ba-3a0ed2d645c0" frameborder="0" allowFullScreen="true"></iframe>
            """
            iframe_html = iframe_html.replace("URL_DU_FICHIER", "https://app.powerbi.com/reportEmbed?reportId=6336e870-43dd-476a-b285-a4456b285cac&autoAuth=true&ctid=48f2d645-8d8d-4cf0-80ba-3a0ed2d645c0")
            st.markdown(iframe_html, unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("https://app.powerbi.com/groups/me/reports/6336e870-43dd-476a-b285-a4456b285cac/ReportSection71f55ee0677c00e11175?bookmarkGuid=b3cb0fe6-44d4-4e65-a793-f1888011af46&bookmarkUsage=1&ctid=48f2d645-8d8d-4cf0-80ba-3a0ed2d645c0&portalSessionId=40616c31-170d-4e97-b6f9-d8e041a757be&fromEntryPoint=export")   

        # PAGE 5 : présentation des meilleurs hyperparamètres de chaque modèle
            
        if page == pages[4]:
            st.sidebar.header("Choix des paramètres pour le modèle")
            st.header("Choix des paramètres pour le modèle")
            model = st.sidebar.selectbox("Recherche des meilleurs paramètres", models)
            st.write("Meilleurs hyperparamètres :", get_param(model))


        # PAGE 6 : entraînement des modèles et test des paramètres

        if page == pages[5]:
            st.header("Entrainement des modèles")
            st.sidebar.header("Entrainement des modèles")
            model = st.sidebar.selectbox("Choisissez votre classificateur", models)
            
            if model == models[0]:
                num_intercept=st.sidebar.radio("Réglage fit_intercept :", [True, False])
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lr2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):

                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, 
                                                                        min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        LinearRegression(fit_intercept=num_intercept))
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)

            if model == models[1]:
                num_algos=st.sidebar.radio("Choisissez algorithm :", ['kd_tree', 'brute', 'auto', 'ball_tree'])
                num_leaf_size=st.sidebar.number_input("Choisissez leaf_size : ", 1, 100, 30)        
                num_metrics=st.sidebar.radio("Choisissez metric :", ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], index=3)
                num_n_neighbors=st.sidebar.number_input("Choisissez  n_neighbors : ", 1, 20, 7)
                num_p=st.sidebar.number_input("Choisissez p : ", 1, 10, 1)  
                num_weights=st.sidebar.radio("Choisissez une fonction de pondération, weights :", ['uniform', 'distance'], index=1)
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_knn2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        KNeighborsRegressor(n_neighbors=num_n_neighbors, weights=num_weights, metric=num_metrics, algorithm=num_algos, 
                                            leaf_size=num_leaf_size, p=num_p))
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)

            if model == models[2]:
                num_n_estimators=st.sidebar.number_input("Choisissez le nombre d'arbres dans la forêt, n_estimators : ", 1, 100, 100)
                num_min_samples_split=st.sidebar.number_input("Choisissez le nombre minimum d'échantillons requis pour diviser un nœud interne, min_samples_split : ", 2, 100, 2)  
                num_min_samples_leaf=st.sidebar.number_input("Choisissez le nombre minimum d'échantillons requis pour être à une feuille, min_samples_leaf : ", 1, 100, 1)  
                num_min_weight_fraction_leaf=st.sidebar.number_input("Choisissez la fraction minimale du poids total des échantillons (pondérés) requise pour être à une feuille, min_weight_fraction_leaf : ", 0.0, 0.5, 0.0)  
                num_max_features=st.sidebar.radio("Choisissez le nombre de caractéristiques à considérer lors de la recherche de la meilleure division, max_features :", ['sqrt', 'log2', 'auto', None], index=2)
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_rf2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, 
                                                                        min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        RandomForestRegressor(n_estimators=num_n_estimators, min_samples_split=num_min_samples_split,
                                        min_samples_leaf=num_min_samples_leaf, min_weight_fraction_leaf=num_min_weight_fraction_leaf, 
                                        max_features=num_max_features))
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae= get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)
                    
            if model == models[3]:
                num_alpha=st.sidebar.number_input("Choisissez le paramètre de régularisation, alpha :", 0.0, 10.0, 0.1)
                num_fit_intercept=st.sidebar.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [True, False])
                num_max_iter=st.sidebar.number_input("Choisissez le nombre maximum d'itérations effectuées :", 0, 5000, 2000)
                num_positive=st.sidebar.radio("Restreint les coefficients à être positifs, positive :", [True, False], index=1)         
                num_precompute=st.sidebar.radio("Utilise une version de précalcul pour la matrice X, precompute :", [True, False])  
                num_selection=st.sidebar.radio("Choisissez la méthode utilisée pour sélectionner les variables dans le modèle :", ['random', 'cyclic'])
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lass2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, 
                                                                        min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        Lasso(alpha=num_alpha, fit_intercept=num_fit_intercept, precompute=num_precompute,
                            max_iter=num_max_iter, positive=num_positive , selection=num_selection))
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)
                
            if model == models[4]:
                num_C=st.sidebar.number_input("Choisissez la force de régularisation, C :", 0.1, 10.0, 0.1)
                num_dual=st.sidebar.radio("Résout le problème dual de la formulation SVM, dual :", [True, False], index=1)  
                num_epsilon=st.sidebar.number_input("Choisissez la marge de tolérance de l'erreur, epsilon :", 0.1, 10.0, 0.1)
                num_fit_intercept=st.sidebar.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [True, False], index=1) 
                num_intercept_scaling=st.sidebar.number_input("Choisissez l'intercept_scaling' :", 0.1, 10.0, 0.1) 
                num_loss=st.sidebar.radio("Choisissez la fonction de perte utilisée pour optimiser le modèle, loss :", ['squared_epsilon_insensitive', 'epsilon_insensitive', 'huber'])
                num_max_iter=st.sidebar.slider("Choisissez le nombre maximum d'itérations effectuées :", 0, 1000, 1000)
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_line2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, 
                                                                        min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        LinearSVR(C=num_C, epsilon=num_epsilon, loss=num_loss, max_iter=num_max_iter,
                                    dual=num_dual, fit_intercept=num_fit_intercept, intercept_scaling=num_intercept_scaling))
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)
            
            if model == models[5]:
                num_cv=st.sidebar.number_input("Nombre de folds pour la validation croisée, CV :", 1, 20, 5)
                num_fit_intercept=st.sidebar.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [True, False])  
                num_max_iter=st.sidebar.number_input("Choisissez le nombre maximum d'itérations effectuées :", 0, 1000, 500)
                num_n_alpha=st.sidebar.number_input("Nombre maximal de valeurs de l'hyperparamètre alpha à tester, n_alphas :", 1, 1000, 1000)
                num_positive=st.sidebar.radio("Choisissez le positive :", [True, False], index=1)
                num_verbose=st.sidebar.radio("Choisissez la verbose :", [True, False], index=1)
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lasso_cv2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        LassoLarsCV(fit_intercept=num_fit_intercept,max_iter=num_max_iter,
                                        cv=num_cv, max_n_alphas=num_n_alpha,
                                        positive=num_positive,verbose=num_verbose))  
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)

            if model == models[6]:
                num_C=st.sidebar.number_input("Choisissez la force de régularisation, C :", 0.1, 100.0, 0.1)
                num_epsilon=st.sidebar.number_input("Choisissez la marge de tolérance de l'erreur, epsilon :", 0.1, 10.0, 0.5)
                num_kernel=st.sidebar.radio("Choisissez le kernel :", ['linear', 'poly', 'rbf', 'sigmoid']) 
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_svr2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):

                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        SVR(C=num_C, epsilon=num_epsilon, kernel=num_kernel))  
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)
            
            if model == models[7]:
                num_max_depth=st.sidebar.number_input("Choisissez max_depth :", 0, 10, 5)
                num_sample_leaf=st.sidebar.number_input("Choisissez min_sample_leaf :", 0, 10, 1)
                num_sample_split=st.sidebar.number_input("Choisissez min_sample_split :", 0, 10, 2)
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_dt2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, 
                                                                        min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        DecisionTreeRegressor(max_depth=num_max_depth, min_samples_leaf=num_sample_leaf,
                                            min_samples_split=num_sample_split))
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)
                
            if model == models[8]:
                num_learning_rate=st.sidebar.number_input("Choisissez learning_rate :", 0.0, 10.0, 0.1)
                num_n_estimators=st.sidebar.number_input("Choisissez n_estimators :", 0, 100, 50)
                graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
                st.subheader('Réglage des paramètres')
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_ab2.joblib')
                st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
                if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
                    pipeline = make_pipeline(
                        StandardScaler(),
                        SelectFwe(alpha=0.009000000000000001),
                        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                        VarianceThreshold(threshold=0.1), 
                        AdaBoostRegressor(learning_rate=num_learning_rate, n_estimators=num_n_estimators))
                    pipeline.fit(X_train, y_train)
                    get_score(model)
                    score_p, y_pred, r2, mse, mae = get_score(model)
                    #st.markdown("Score : ")
                    #st.write(score_p)
                    st.markdown("R2 : ")
                    st.write(r2)
                    st.markdown("MSE : ")
                    st.write(mse)
                    st.markdown("MAE : ")
                    st.write(mae)
                    plot_perf(graphe_perf)


        # PAGE 7 : présentation de l'interprétabilité du modèle grâce à la 
        # visualisation des variables les plus exploitées pour cahque modèle
        
        if page == pages[6]:
            st.header("Interprétabilité du modèle")
            model = st.selectbox("Choisissez votre modèle", models)
            
            if model == models[0]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lr2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1))
                pipeline.fit(X_train, y_train)
                lr=LinearRegression(**best_params)
                lr.fit(X_train, y_train)
                
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                coefficients = pd.Series(lr.coef_, index=X_train.columns)
                top_15_features = coefficients.abs().nlargest(15) 
                fig=plt.figure(figsize=(10, 6))
                top_15_features.plot(kind='barh')
                plt.xlabel('Coefficient')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (Linear Regression)')
                st.pyplot(fig)

            if model == models[1]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_knn2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1))
                pipeline.fit(X_train, y_train)
                knn=KNeighborsRegressor(**best_params)
                knn.fit(X_train, y_train)
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                sorted_importances=load("C:/Users/'+username+'/Documents/Projet_DA/Streamlit/shap/sorted_importances_knn.joblib")
                sorted_feature_names=load("C:/Users/'+username+'/Documents/Projet_DA/Streamlit/shap/sorted_feature_names_knn.joblib")
                top_feature_names = sorted_feature_names[:15]
                top_importances = sorted_importances[:15]
                fig=plt.figure(figsize=(10, 6))
                plt.barh(range(len(top_importances)), top_importances, tick_label=top_feature_names)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (KNN)')
                st.pyplot(fig)
                
            if model == models[2]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_rf2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1))
                pipeline.fit(X_train, y_train)
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                rf=RandomForestRegressor(**best_params)
                rf.fit(X_train, y_train)
                explainer = shap.TreeExplainer(rf)
                shap_values = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/shap/shap_values_rf.joblib')
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(shap.summary_plot(shap_values, X_test, plot_type="bar"))
                
                st.pyplot(shap.summary_plot(shap_values, X_test))
                        
            if model == models[3]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lass2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1)
                    )
                pipeline.fit(X_train, y_train)
                lass=Lasso(**best_params)
                lass.fit(X_train, y_train)
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                coefficients = pd.Series(lass.coef_, index=X_train.columns)
                top_15_features = coefficients.abs().nlargest(15) 
                fig=plt.figure(figsize=(10, 6))
                top_15_features.plot(kind='barh')
                plt.xlabel('Coefficient')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (Lasso)')
                st.pyplot(fig)    
                    
            if model == models[4]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_line2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1)
                    )
                pipeline.fit(X_train, y_train)
                line=LinearSVR(**best_params)
                line.fit(X_train, y_train)
                
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                coefficients = pd.Series(line.coef_, index=X_train.columns)
                top_15_features = coefficients.abs().nlargest(15) 
                fig=plt.figure(figsize=(10, 6))
                top_15_features.plot(kind='barh')
                plt.xlabel('Coefficient')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (LinearSVR)')
                st.pyplot(fig)    
                
            if model == models[5]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_lasso_cv2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1))  
                pipeline.fit(X_train, y_train)
                lassCV=LassoLarsCV(**best_params)
                lassCV.fit(X_train, y_train)
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                coefficients = pd.Series(lassCV.coef_, index=X_train.columns)
                top_15_features = coefficients.abs().nlargest(15) 
                fig=plt.figure(figsize=(10, 6))
                top_15_features.plot(kind='barh')
                plt.xlabel('Coefficient')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (LassoLarsCV)')
                st.pyplot(fig) 
                
            if model == models[6]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_svr2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, 
                                                                    min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1) 
                    )  
                pipeline.fit(X_train, y_train)
                svr=SVR(**best_params)
                svr.fit(X_train, y_train)
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                sorted_importances=load("C:/Users/"+username+"/Documents/Projet_DA/Streamlit/shap/sorted_importances_svr.joblib")
                sorted_feature_names=load("C:/Users/"+username+"/Documents/Projet_DA/Streamlit/shap/sorted_feature_names_svr.joblib")
                top_feature_names = sorted_feature_names[:15]
                top_importances = sorted_importances[:15]
                fig=plt.figure(figsize=(10, 6))
                plt.barh(range(len(top_importances)), top_importances, tick_label=top_feature_names)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (SVR)')
                st.pyplot(fig)
                
            if model == models[7]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_dt2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1))
                pipeline.fit(X_train, y_train)
                dt=DecisionTreeRegressor(**best_params)
                dt.fit(X_train, y_train)
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                sorted_importances=load("C:/Users/"+username+"/Documents/Projet_DA/Streamlit/shap/sorted_importances_dt.joblib")
                sorted_feature_names=load("C:/Users/"+username+"/Documents/Projet_DA/Streamlit/shap/sorted_feature_names_dt.joblib")
                top_feature_names = sorted_feature_names[:15]
                top_importances = sorted_importances[:15]
                fig=plt.figure(figsize=(10, 6))
                plt.barh(range(len(top_importances)), top_importances, tick_label=top_feature_names)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (DecisionTreeRegressor)')
                st.pyplot(fig)
                
            if model == models[8]:
                best_params = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/best_params/best_params_ab2.joblib')
                pipeline = make_pipeline(
                    StandardScaler(),
                    SelectFwe(alpha=0.009000000000000001),
                    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, 
                                                                    max_features=0.25, min_samples_leaf=14, 
                                                                    min_samples_split=7, n_estimators=100)),
                    VarianceThreshold(threshold=0.1))
                pipeline.fit(X_train, y_train)
                ab=AdaBoostRegressor(**best_params)
                ab.fit(X_train, y_train)
                st.markdown("Voici les variables qui ont le plus d'impact dans les décisions prises par le modèle.")
                sorted_importances=load("C:/Users/"+username+"/Documents/Projet_DA/Streamlit/shap/sorted_importances_ab.joblib")
                sorted_feature_names=load("C:/Users/"+username+"/Documents/Projet_DA/Streamlit/shap/sorted_feature_names_ab.joblib")
                top_feature_names = sorted_feature_names[:15]
                top_importances = sorted_importances[:15]
                fig=plt.figure(figsize=(10, 6))
                plt.barh(range(len(top_importances)), top_importances, tick_label=top_feature_names)
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Top 15 Feature Importances (AdaBoostRegressor)')
                st.pyplot(fig)

        # PAGE 8 : Test du modèle avec les données du dataframe créé pour tester le modèle
                
        if page == pages[7]:
            st.header("Test du modèle")
            st.subheader("Nous allons maintenant prédire les ventes des jeux suivants :")

            edited_df=st.data_editor(df_new_data, num_rows='dynamic') 
            #st.markdown("Appliquez mon modèle de ML à ces nouvelles données:")
            #new_X_test=st.experimental_data_editor(X_test, num_rows='dynamic') 
            rf = load('C:/Users/'+username+'/Documents/Projet_DA/Streamlit/rf.joblib')
            y_pred=rf.predict(X_test)
            
            st.subheader("Hogwarts Legacy")
            st.image("hogwarts.jpeg")
            st.write("Notre prédiction de vente s'élève à",round(y_pred[-5],1)*100, "millions d'exemplaires.")
            st.write("Le jeu s'est vendu à 15 millions d'exemplaires.")
            st.write("")
            st.write("Notre modèle a surévalué les ventes de ce jeu.")

            st.subheader("It Takes Two")
            st.image("it2.jpeg")
            st.write("Notre prédiction de vente s'élève à ",round(y_pred[-4],1)*100, "millions d'exemplaires.")
            st.write("Le jeu s'est vendu à 10 millions d'exemplaires.")
            st.write("")
            st.write("Notre prédiction est assez efficace.") 
            
            st.subheader("Resident Evil: Village")
            st.image("village.png")
            st.write("Notre prédiction de vente s'élève à",round(y_pred[-3],1)*100, "millions d'exemplaires.")
            st.write("Le jeu s'est vendu à 8 millions d'exemplaires.")
            st.write("")
            st.write("Notre modèle a surévalué les ventes de ce jeu.")   

            st.subheader("Hitman 2")
            st.image("hit.jpg")
            st.write("Notre prédiction de vente s'élève à",round(y_pred[-2],1)*100, "millions d'exemplaires.")
            st.write("Le jeu s'est vendu à 8 millions d'exemplaires.")
            st.write("")
            st.write("Notre prédiction est assez efficace.") 
            
            st.subheader("Animal Crossing : New Horizons")
            st.image("ac.jpg")
            st.write("Notre prédiction de vente s'élève à",round(y_pred[-1],1)*100, "millions d'exemplaires.")
            st.write("Le jeu s'est vendu à 32.63 millions d'exemplaires.")
            st.write("")
            st.write("Notre modèle a surévalué les ventes de ce jeu.") 
            
            st.header("CONCLUSION")
            st.write("")
            st.write("En conclusion, le modèle a tendance a surévaluer les ventes. Toutefois il reste assez satisfaisant et les tests démontrent tout à fait son score de prédiction de 0.63.")            
                         
    else:
        st.error("Mot de passe incorrect")
 
if __name__ == '__main__':
    main()                