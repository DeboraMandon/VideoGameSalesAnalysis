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
import scikitplot as skplt

@st.cache_data
def load_data():
    df= pd.read_csv("gaming_total_v2.csv")
    df= df.drop('Unnamed: 0', axis=1)
    df_clean=pd.read_csv("df_clean_v2.csv", index_col=0)
    df_gd=pd.get_dummies(df_clean)
    return df, df_clean, df_gd
df, df_clean, df_gd = load_data()

data_n = {
    "Platform": ["Xbox Series"],
    "Year": [2023],
    "Genre": ["Action"],
    "Publisher": ["Warner Bros Games"],
    "Meta_NP": [14],
    "Meta_VP": [43],
    "Meta_NUsers": [130],
    "Meta_VUsers": [2.3],
    "test_note_JVC": [18],
    "avis_count_JVC": [61],
    "avis_note_JVC": [15.3],
    "Classification_Age_JVC": [8],
    "Support_JVC": ["DVD"],
    "langue_parlée_JVC": ["français"],
    "texte_JVC": ["français"]}
df_new_data = pd.DataFrame(data_n)

pages=['📖 Présentation du projet', '🗃️ Dataframe', '📈 Data Visualisation', 
       '🛠️ Hyperparamètres', '🚀 Modélisation', '🪄 Test du modèle']
models= ["Regression Linéaire", "KNN", "Random forest", 'Lasso', 
         'LinearSVR', 'LassoLarsCV', 'SVR', 'DecisionTreeRegressor', 'AdaBoostRegressor']
graphes=["Graphique de régression", "Cumulative Gains Curve"]
graphs=["Evolution des ventes par Région", "Répartition des Ventes par Région", 
            "Répartition des catégories par variables catégorielles", #"Distribution des variables numériques",
            "Distribution des variables", "Heatmap des variables numériques du Dataframe après PreProcessing"]

st.sidebar.image('DS.jpg')
st.sidebar.title("Video Games Sales Analysis")
page=st.sidebar.radio("Choisissez votre page",pages)
#st.sidebar.header("Machine Learning Project")
#st.sidebar.subheader("Auteur : Débora Mandon")
#st.sidebar.subheader("Navigation")

# Entrainement des modèles
y=df_gd['Global_Sales']
X=df_gd.drop('Global_Sales', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

def get_param(model):
    if model == models[0]:
        best_params_lr = load('best_params_lr2.joblib') 
        best_score = load('best_score_lr2.joblib')
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
        best_params_knn = load('best_params_knn2.joblib') 
        best_score = load('best_score_knn2.joblib')
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
        best_params_rf = load('best_params_rf2.joblib') 
        best_score = load('best_score_rf2.joblib')
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
        best_params_lass = load('best_params_lass2.joblib') 
        best_score = load('best_score_lass2.joblib')
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
        best_params_line = load('best_params_line2.joblib') 
        best_score = load('best_score_line2.joblib')
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
        best_params_lasso_cv = load('best_params_lasso_cv2.joblib') 
        best_score = load('best_score_lasso_cv2.joblib')
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
        best_params_svr = load('best_params_svr2.joblib') 
        best_score = load('best_score_svr2.joblib')
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
        best_params_dt = load('best_params_dt2.joblib') 
        best_score = load('best_score_dt2.joblib')
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
        best_params_ab = load('best_params_ab2.joblib') 
        best_score = load('best_score_ab2.joblib')
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
                
def get_score(model):
    score_p = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return(score_p, y_pred, r2, mse, mae)
    
def plot_perf(graphe_perf):
    if graphe_perf == graphes[0]:
        fig1=plt.figure()
        plt.scatter(y_test, y_pred, color='darkblue')
        plt.xlabel('Vraies valeurs')
        plt.ylabel('Prédictions')
        plt.title('Graphique de régression')
        st.pyplot(fig1)
    
    #if graphe_perf == graphes[1]:


if page == pages[0]:
    st.image('image-jeux-video.jpg', use_column_width=1)
    st.title("Video Games Sales Analysis")
    st.header("Machine Learning Project")
    st.subheader("Auteur : Débora Mandon")
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
    st.sidebar.subheader("Dataframe")
    if st.sidebar.checkbox("Afficher les données brutes :", False):
        st.subheader("Jeu de données 'vg_sales' : Echantillon de 100 observations")
        st.write(df.sample(100))
    
    st.header("Analyse du Dataframe")
    #st.image("analytics.jpg")
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
    st.write("Nom des colonnes :",df.columns.to_list())

    st.markdown("")
    #st.dataframe(df.head(10))
    st.markdown("")
    st.write("Shape du Dataframe :",df.shape)
    #st.write("Nom des colonnes :",df.columns)
    st.write("Description du Dataframe :",df.describe())
    st.markdown("")
    st.markdown("Pour obtenir le jeu de données utilisé pour le Machine Learning, nous avons réalisé une étape \n"
                "de Pre Processing pour traiter les valeurs nulles, les variables très corrélées, supprimer les variables \n"
                "qui n'apportent aucune information... Nous avons également utilisé des méthode de Text Mining \n"
                "pour extraire les informations utiles des chaînes de caractères scrppés sur Internet.")
    st.markdown("")
    st.header("Le Dataframe final")    
    st.write("Shape du nouveau Dataframe.v2 : ", df_clean.shape)
    st.write("Nom des colonnes :",df_clean.columns.to_list())
    st.write("Description du Dataframe.v2 :",df_clean.describe(), df_clean.describe(include='object'))    

if page == pages[2]:
    st.header("Data Visualisation")
    st.sidebar.header("Data Visualisation")
    #st.subheader("Visualiser les données à l'aide de graphiques, choisissez un type de visualisation puis les variables à explorer.")
    graph=st.sidebar.selectbox("Choisissez votre visualisation", graphs)

    if graph == graphs[0]:    
        for i in df.select_dtypes(include=['int64', 'float64']):
            df[f'cat_{i}'] = pd.qcut(df[i], q=[0,.25,.5,.75,1.], duplicates='drop')
        df['Year']=pd.to_datetime(df['Year'], format='%Y')
        sales_per_year=df.groupby('Year', as_index=False).agg({'NA_Sales':sum, 'EU_Sales':sum, 'JP_Sales':sum, 'Other_Sales':sum,'Global_Sales':sum})
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
        choices_cat =st.sidebar.radio("Choisissez les variables catégorielles à étudier", df_clean.select_dtypes(include=['object']).columns)
        fig3=plt.figure()
        df_clean.select_dtypes(include=['object'])[choices_cat].value_counts().plot.pie(autopct=lambda x: f'{str(round(x, 2))}%')
        plt.title(f'Répartition de la variable {str(choices_cat)}')
        st.pyplot(fig3)

    #if graph == graphs[3]:
    #    choices_num =st.sidebar.radio("Choisissez les variables numériques à étudier", df_clean.select_dtypes(include=['int64', 'float64']).columns[:-1])
    #    fig4=plt.figure()
    #    sns.distplot(df[choices_num], label=choices_num)
    #    plt.title(f'Histogramme de la variable {str(choices_num)}')
    #    plt.legend(loc='best')
    #    st.pyplot(fig4)

    if graph == graphs[3]:     
        choices_var =st.sidebar.radio("Choisissez les variables à étudier", df_clean.columns)
        fig5=plt.figure()
        sns.histplot(df_clean[choices_var], label=choices_var)
        #sns.histplot(df_le['Global_Sales'])
        plt.xticks(rotation=70, ha='right', fontsize=8)
        plt.legend()
        st.pyplot(fig5)
            
    if graph == graphs[4]:    
        fig6=plt.figure()
        sns.heatmap(df_clean.select_dtypes(include=['int64', 'float64']).corr(),annot=False)   
        st.pyplot(fig6)
       
if page == pages[3]:
    st.sidebar.header("Choix des paramètres pour le modèle")
    st.header("Choix des paramètres pour le modèle")
    model = st.sidebar.selectbox("Recherche des meilleurs paramètres", models)
    st.write("Meilleurs hyperparamètres :", get_param(model))
    #st.image('engrenage.gif')

if page == pages[4]:
    st.header("Entrainement des modèles")
    st.sidebar.header("Entrainement des modèles")
    model = st.sidebar.selectbox("Choisissez votre classificateur", models)
    #st.image('ml.gif')
    
    if model == models[0]:
        num_intercept=st.sidebar.radio("Réglage fit_intercept :", [True, False])
        #num_copy_X=st.sidebar.radio("Choisissez copy_X :", [True, False])
        #num_n_jobs=st.sidebar.number_input("Choisissez n_jobs : ", -1, 10, 1)
        #num_positive=st.sidebar.radio("Choisissez positive :", [True, False], index=1)
        graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
        st.subheader('Réglage des paramètres')
        #st.image("ml.jpg")
        best_params = load('best_params_lr2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):

            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                LinearRegression(fit_intercept=num_intercept))
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
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
        #st.image("ml.gif")
        best_params = load('best_params_knn2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                        alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                KNeighborsRegressor(n_neighbors=num_n_neighbors, weights=num_weights, metric=num_metrics, algorithm=num_algos, 
                                    leaf_size=num_leaf_size, p=num_p))
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
            st.markdown("R2 : ")
            st.write(r2)
            st.markdown("MSE : ")
            st.write(mse)
            st.markdown("MAE : ")
            st.write(mae)
            plot_perf(graphe_perf)

    if model == models[2]:
        num_n_estimators=st.sidebar.number_input("Choisissez le nombre d'arbres dans la forêt, n_estimators : ", 1, 100, 100)
        #num_criterion=st.sidebar.radio("Choisissez la mesure de qualité de la division de l'arbre, criterion :", ['friedman_mse', 'squared_error', 'absolute_error', 'poisson'])
        num_min_samples_split=st.sidebar.number_input("Choisissez le nombre minimum d'échantillons requis pour diviser un nœud interne, min_samples_split : ", 2, 100, 2)  
        num_min_samples_leaf=st.sidebar.number_input("Choisissez le nombre minimum d'échantillons requis pour être à une feuille, min_samples_leaf : ", 1, 100, 1)  
        num_min_weight_fraction_leaf=st.sidebar.number_input("Choisissez la fraction minimale du poids total des échantillons (pondérés) requise pour être à une feuille, min_weight_fraction_leaf : ", 0.0, 0.5, 0.0)  
        num_max_features=st.sidebar.radio("Choisissez le nombre de caractéristiques à considérer lors de la recherche de la meilleure division, max_features :", ['sqrt', 'log2', 'auto', None], index=2)
        graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
        st.subheader('Réglage des paramètres')
        #st.image("ml.gif")
        best_params = load('best_params_rf2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                RandomForestRegressor(n_estimators=num_n_estimators, min_samples_split=num_min_samples_split,
                                   min_samples_leaf=num_min_samples_leaf, min_weight_fraction_leaf=num_min_weight_fraction_leaf, 
                                   max_features=num_max_features))
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae= get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
            st.markdown("R2 : ")
            st.write(r2)
            st.markdown("MSE : ")
            st.write(mse)
            st.markdown("MAE : ")
            st.write(mae)
            plot_perf(graphe_perf)
              
    if model == models[3]:
        num_alpha=st.sidebar.number_input("Choisissez le paramètre de régularisation, alpha :", 0.0, 10.0, 0.1)
        #num_copy_X=st.sidebar.radio("Choisissez copy_X :", [True, False], index=1)      
        num_fit_intercept=st.sidebar.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [True, False])
        num_max_iter=st.sidebar.number_input("Choisissez le nombre maximum d'itérations effectuées :", 0, 5000, 2000)
        num_positive=st.sidebar.radio("Restreint les coefficients à être positifs, positive :", [True, False], index=1)         
        num_precompute=st.sidebar.radio("Utilise une version de précalcul pour la matrice X, precompute :", [True, False])  
        num_selection=st.sidebar.radio("Choisissez la méthode utilisée pour sélectionner les variables dans le modèle :", ['random', 'cyclic'])
        graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
        st.subheader('Réglage des paramètres')
        #st.image("ml.gif")
        best_params = load('best_params_lass2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                Lasso(alpha=num_alpha, fit_intercept=num_fit_intercept, precompute=num_precompute,
                      max_iter=num_max_iter, positive=num_positive , selection=num_selection))
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
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
        #st.image("ml.gif")
        best_params = load('best_params_line2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                LinearSVR(C=num_C, epsilon=num_epsilon, loss=num_loss, max_iter=num_max_iter,
                            dual=num_dual, fit_intercept=num_fit_intercept, intercept_scaling=num_intercept_scaling))
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
            st.markdown("R2 : ")
            st.write(r2)
            st.markdown("MSE : ")
            st.write(mse)
            st.markdown("MAE : ")
            st.write(mae)
            plot_perf(graphe_perf)
     
    if model == models[5]:
        #num_copy_X=st.sidebar.radio("Choisissez copy_X :", [True, False])      
        num_cv=st.sidebar.number_input("Nombre de folds pour la validation croisée, CV :", 1, 20, 5)
        num_fit_intercept=st.sidebar.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [True, False])  
        num_max_iter=st.sidebar.number_input("Choisissez le nombre maximum d'itérations effectuées :", 0, 1000, 500)
        num_n_alpha=st.sidebar.number_input("Nombre maximal de valeurs de l'hyperparamètre alpha à tester, n_alphas :", 1, 1000, 1000)
        #num_n_jobs=st.sidebar.slider("Choisissez le nombre de n_jobs : ", -1, 30, 23)        
        #num_normalize=st.sidebar.radio("Normalise les variables explicatives, normalize :", [False, True])  
        num_positive=st.sidebar.radio("Choisissez le positive :", [True, False], index=1)
        #num_precompute=st.sidebar.radio("Utilise une version de précalcul pour la matrice X, precompute :", [True, False])  
        num_verbose=st.sidebar.radio("Choisissez la verbose :", [True, False], index=1)
        graphe_perf=st.sidebar.selectbox("Choisissez un graphique de performance du modèle ML :", graphes)
        st.subheader('Réglage des paramètres')
        #st.image("ml.gif")
        best_params = load('best_params_lasso_cv2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                LassoLarsCV(fit_intercept=num_fit_intercept,max_iter=num_max_iter, #normalize=num_normalize, precompute=num_precompute,
                                cv=num_cv, max_n_alphas=num_n_alpha,#n_jobs=num_n_jobs, 
                                positive=num_positive,verbose=num_verbose))  
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
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
        #st.image("ml.gif")
        best_params = load('best_params_svr2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):

            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                SVR(C=num_C, epsilon=num_epsilon, kernel=num_kernel))  
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
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
        #st.image("ml.gif")
        best_params = load('best_params_dt2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                DecisionTreeRegressor(max_depth=num_max_depth, min_samples_leaf=num_sample_leaf,
                                      min_samples_split=num_sample_split))
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
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
        #st.image("ml.gif")
        best_params = load('best_params_ab2.joblib')
        st.write('Rappel des meilleurs hyperparamètres de la GridSearchCV pour le modèle:', best_params)
        if st.button('Execution du modèle avec les réglages sélectionnés', key="classify"):
            pipeline = make_pipeline(
                StandardScaler(),
                #MinMaxScaler(),
                SelectFwe(#score_func=f_regression, 
                          alpha=0.009000000000000001),
                StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
                VarianceThreshold(threshold=0.1), 
                AdaBoostRegressor(learning_rate=num_learning_rate, n_estimators=num_n_estimators))
            #set_param_recursive(pipeline.steps, 'random_state', 42)
            pipeline.fit(X_train, y_train)
            get_score(model)
            score_p, y_pred, r2, mse, mae = get_score(model)
            st.markdown("Score : ")
            st.write(score_p)
            st.markdown("R2 : ")
            st.write(r2)
            st.markdown("MSE : ")
            st.write(mse)
            st.markdown("MAE : ")
            st.write(mae)
            plot_perf(graphe_perf)
   
if page == pages[5]:
    st.header("Test du modèle")

    edited_df=st.experimental_data_editor(df_new_data, num_rows='dynamic') 
    st.markdown("Appliquez mon modèle de ML à ces nouvelles données:")
    
    new_data_encoded = pd.get_dummies(edited_df)
    values_test=list(edited_df.iloc[0])
    st.write(values_test)
        
    #if st.button('Execution', key="classify"):

        