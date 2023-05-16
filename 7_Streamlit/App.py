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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import re
import joblib
from joblib import dump, load

df= pd.read_csv("gaming_total_v2.csv")
df= df.drop('Unnamed: 0', axis=1)
df_clean=pd.read_csv("df_clean_v2.csv", index_col=0)
df_gd=pd.get_dummies(df_clean)



st.sidebar.title("Video Games Sales Analysis")
st.sidebar.image('image-jeux-video.jpg')
st.sidebar.subheader("Navigation")

pages=['Présentation du projet', 'Dataframe', 'Data Visualisation', 'Hyperparamètres', 'Modélisation']
models= ["Regression Linéaire", "KNN", "Random forest", 'Lasso', 'LinearSVR', 'LassoLarsCV', 'SVR', 'DecisionTreeRegressor', 'AdaBoostRegressor']
page=st.sidebar.radio("Choisissez votre page", pages)


if page == pages[0]:
    st.title("Video Games Sales Analysis")
    st.header("Machine Learning Project")
    st.subheader("par Débora Mandon")
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
    st.subheader("Visualiser les données à l'aide de graphiques, choisissez un type de visualisation puis les variables à explorer."
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
        plt.title(f'Répartition de la variable {str(choices_cat)}')
        st.pyplot(fig3)

    if graph == graphs[3]:
        choices_num =st.radio("Choisissez les variables numériques à étudier", df_clean.select_dtypes(include=['int64', 'float64']).columns[:-1])
        fig4=plt.figure()
        sns.distplot(df[choices_num], label=choices_num)
        plt.title(f'Histogramme de la variable {str(choices_num)}')
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


def get_param(model):
    if model == models[0]:
        best_params_lr = load('best_params_lr.joblib') 
        best_score = load('best_score_lr.joblib')
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
        best_params_knn = load('best_params_knn.joblib') 
        best_score = load('best_score_knn.joblib')
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
        best_params_rf = load('best_params_rf.joblib') 
        best_score = load('best_score_rf.joblib')
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
        best_params_lass = load('best_params_lass.joblib') 
        best_score = load('best_score_lass.joblib')
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
        best_params_line = load('best_params_line.joblib') 
        best_score = load('best_score_line.joblib')
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
        best_params_lasso_cv = load('best_params_lasso_cv.joblib') 
        best_score = load('best_score_lasso_cv.joblib')
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
        best_params_svr = load('best_params_svr.joblib') 
        best_score = load('best_score_svr.joblib')
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
        best_params_dt = load('best_params_dt.joblib') 
        best_score = load('best_score_dt.joblib')
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
        best_params_ab = load('best_params_ab.joblib') 
        best_score = load('best_score_ab.joblib')
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
    if model == models[0]:
        num_intercept=st.radio("Choisissez un fit_intercept :", [True, False])
        num_copy_X=st.radio("Choisissez copy_X :", [True, False])
        num_n_jobs=st.slider("Choisissez le nombre de n_jobs : ", -1, 10, -1)
        num_positive=st.radio("Choisissez le positive :", [False, True])
        
        best_params = load('best_params_lr.joblib')
        
        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        LinearRegression(fit_intercept=num_intercept, copy_X=num_copy_X, n_jobs=num_n_jobs, positive=num_positive))
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)

                
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Linear Regression Performance')
        st.pyplot(fig)
                
        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae
        
    if model == models[1]:
        
        num_n_neighbors=st.slider("Choisissez le nombre de voisins, n_neighbors : ", 1, 50, 11)
        num_weights=st.radio("Choisissez une fonction de pondération, weights :", ['uniform', 'distance'])
        num_metrics=st.radio("Choisissez une fonction de distance entre les points, metric :", ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
        num_algos=st.radio("Choisissez une fonction de distance entre les points, algorithm :", ['kd_tree', 'brute', 'auto', 'ball_tree'])
        num_leaf_size=st.slider("Choisissez taille de la feuille de l'arbre, leaf_size : ", 1, 100, 1)        
        num_p=st.slider("Choisissez la taille de p, p : ", 1, 10, 1)  
                
        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        KNeighborsRegressor(n_neighbors=num_n_neighbors, weights=num_weights, metric=num_metrics, algorithm=num_algos, 
                                leaf_size=num_leaf_size, p=num_p))
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)
        
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('KNeighborsRegressor Performance \n R-squared score: {:.2f}'.format(r2))
        st.pyplot(fig)
        
        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae  

    if model == models[2]:
 
        num_n_estimators=st.slider("Choisissez nombre d'arbres dans la forêt, n_estimators : ", 1, 100, 18)
        num_criterion=st.radio("Choisissez la mesure de qualité de la division de l'arbre, criterion :", ['friedman_mse', 'squared_error', 'absolute_error', 'poisson'])
        num_min_samples_split=st.slider("Choisissez le nombre minimum d'échantillons requis pour diviser un nœud interne, min_samples_split : ", 2, 100, 100)  
        num_min_samples_leaf=st.slider("Choisissez le nombre minimum d'échantillons requis pour être à une feuille, min_samples_leaf : ", 1, 100, 100)  
        num_min_weight_fraction_leaf=st.slider("Choisissez la fraction minimale du poids total des échantillons (pondérés) requise pour être à une feuille, min_weight_fraction_leaf : ", 0.0, 0.5, 0.5)  
        num_max_features=st.radio("Choisissez le nombre de caractéristiques à considérer lors de la recherche de la meilleure division, max_features :", ['sqrt', 'log2', 'auto', None])

        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        RandomForestRegressor(n_estimators=num_n_estimators, criterion=num_criterion, min_samples_split=num_min_samples_split,
                                   min_samples_leaf=num_min_samples_leaf, min_weight_fraction_leaf=num_min_weight_fraction_leaf, 
                                   max_features=num_max_features))
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)
        
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('RandomForestRegressor Performance \n R-squared score: {:.2f}'.format(r2))
        st.pyplot(fig)        
        
        
        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        fig2=plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation='vertical')
        plt.xlim([-1, X_train.shape[1]])
        st.pyplot(fig2)
        
        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae  
    
    if model == models[3]:
        num_alpha=st.slider("Choisissez le paramètre de régularisation, alpha :", 0.0, 20.0, 17.0)
        num_fit_intercept=st.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [False, True])
        num_precompute=st.radio("Utilise une version de précalcul pour la matrice X, precompute :", [True, False])  
        num_copy_X=st.radio("Choisissez copy_X :", [True, False])      
        num_max_iter=st.slider("Choisissez le nombre maximum d'itérations effectuées :", 0, 10000, 188)
        num_positive=st.radio("Restreint les coefficients à être positifs, positive :", [True, False])         
        num_selection=st.radio("Choisissez la méthode utilisée pour sélectionner les variables dans le modèle :", ['random', 'cyclic'])

        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        Lasso(alpha=num_alpha, fit_intercept=num_fit_intercept, precompute=num_precompute, 
                   copy_X=num_copy_X, max_iter=num_max_iter, positive=num_positive , selection=num_selection))
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)
        
        fig=plt.figure()        
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Lasso Performance \n R-squared score: {:.2f}'.format(r2))
        st.pyplot(fig)

        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae  

    if model == models[4]:
                      
        num_max_iter=st.slider("Choisissez le nombre maximum d'itérations effectuées :", 0, 1000, 543)
        num_C=st.slider("Choisissez la force de régularisation, C :", 0.1, 1000.0, 2.8)
        num_epsilon=st.slider("Choisissez la marge de tolérance de l'erreur, epsilon :", 0.1, 1000000.0, 1.9)
        num_loss=st.radio("Choisissez la fonction de perte utilisée pour optimiser le modèle, loss :", ['squared_epsilon_insensitive', 'epsilon_insensitive', 'huber'])
        num_dual=st.radio("Résout le problème dual de la formulation SVM, dual :", [False, True])  
        num_fit_intercept=st.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [False, True])  

        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        LinearSVR(C=num_C, epsilon=num_epsilon, loss=num_loss, max_iter=num_max_iter,
                       dual=num_dual, fit_intercept=num_fit_intercept))
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)         
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)
        
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('LinearSVR Performance \n R-squared score: {:.2f}'.format(r2))
        st.pyplot(fig)

        # Create the residuals plot
        residuals = y_test - y_pred
        fig2=plt.figure()
        plt.scatter(y_pred, residuals)
        plt.plot([min(y_pred), max(y_pred)], [0, 0], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        st.pyplot(fig2)

        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae 

    if model == models[5]:
        num_fit_intercept=st.radio("Calcule l'ordonnée à l'origine, fit_intercept :", [False, True])  
        num_max_iter=st.slider("Choisissez le nombre maximum d'itérations effectuées :", 0, 1000, 181)
        num_normalize=st.radio("Normalise les variables explicatives, normalize :", [False, True])  
        num_precompute=st.radio("Utilise une version de précalcul pour la matrice X, precompute :", [True, False])  
        num_cv=st.slider("Nombre de folds pour la validation croisée, CV :", 1, 20, 15)
        num_n_alpha=st.slider("Nombre maximal de valeurs de l'hyperparamètre alpha à tester, n_alphas :", 1, 1000, 151)
        num_n_jobs=st.slider("Choisissez le nombre de n_jobs : ", -1, 30, 23)        
        num_copy_X=st.radio("Choisissez copy_X :", [True, False])      

        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        LassoLarsCV(fit_intercept=num_fit_intercept,max_iter=num_max_iter, normalize=num_normalize, precompute=num_precompute,
                           cv=num_cv, max_n_alphas=num_n_alpha,n_jobs=num_n_jobs, copy_X=num_copy_X))
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)                  
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)
        
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('LassoLarsCV Performance \n R-squared score: {:.2f}'.format(r2))
        st.pyplot(fig)

        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae  
    
    if model == models[6]:
        #num_intercept=st.radio("Choisissez un fit_intercept :", [True, False])
        #num_copy_X=st.radio("Choisissez copy_X :", [True, False])
        #num_n_jobs=st.slider("Choisissez le nombre de n_jobs : ", -1, 10, -1)
        #num_positive=st.radio("Choisissez le positive :", [False, True])
        
        best_params = load('best_params_svr.joblib')
        
        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        SVR())
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)

                
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Linear Regression Performance')
        st.pyplot(fig)
                
        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae, "best params :", best_params  
    
    if model == models[7]:
        #num_intercept=st.radio("Choisissez un fit_intercept :", [True, False])
        #num_copy_X=st.radio("Choisissez copy_X :", [True, False])
        #num_n_jobs=st.slider("Choisissez le nombre de n_jobs : ", -1, 10, -1)
        #num_positive=st.radio("Choisissez le positive :", [False, True])
        
        best_params = load('best_params_dt.joblib')
        
        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        DecisionTreeRegressor())
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)

                
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Linear Regression Performance')
        st.pyplot(fig)
                
        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae
    
    if model == models[8]:
        #num_intercept=st.radio("Choisissez un fit_intercept :", [True, False])
        #num_copy_X=st.radio("Choisissez copy_X :", [True, False])
        #num_n_jobs=st.slider("Choisissez le nombre de n_jobs : ", -1, 10, -1)
        #num_positive=st.radio("Choisissez le positive :", [False, True])
        
        best_params = load('best_params_ab.joblib')
        
        pipeline = make_pipeline(
        StandardScaler(),
        MinMaxScaler(),
        SelectFwe(score_func=f_regression, alpha=0.009000000000000001),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=14, min_samples_split=7, n_estimators=100)),
        VarianceThreshold(threshold=0.1), 
        AdaBoostRegressor())
        
        #set_param_recursive(pipeline.steps, 'random_state', 42)
        pipeline.fit(X_train, y_train)
        score_p = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.markdown("Score : ")
        st.write(score_p)
        st.markdown("R2 : ")
        st.write(r2)
        st.markdown("MSE : ")
        st.write(mse)
        st.markdown("MAE : ")
        st.write(mae)

                
        fig=plt.figure()
        plt.scatter(y_pred, y_test)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Linear Regression Performance')
        st.pyplot(fig)
                
        return "Score :",score_p, "R2 :",r2, "MSE :",mse, "MAE :",mae  

if page == pages[3]:
    st.header("Choix des paramètres pour le modèle")
    st.image('param.jpg')
    model = st.selectbox("Recherche des meilleurs paramètres", models)
    st.write("Meilleurs hyperparamètres :", get_param(model))


if page == pages[4]:
    st.title("Entrainement des modèles")
    st.header("Choix du modèle")
    model = st.selectbox("Choisissez votre modèle", models)
    st.image('ml.jpg')
    if model == models[0]:
        best_params = load('best_params_lr.joblib')
        st.write('Meilleurs hyperparamètres pour le modèle:', best_params)
        st.header('Réglage des paramètres')
        st.write("Scores obtenu :", get_score(model))

    if model == models[1]:
        best_params = load('best_params_knn.joblib')
        st.write('Meilleurs hyperparamètres pour le modèle:', best_params)
        st.header('Réglage des paramètres')
        st.write("Scores obtenu :", get_score(model))

    if model == models[2]:
        best_params = load('best_params_rf.joblib')
        st.write('Meilleurs hyperparamètres pour le modèle:', best_params)
        st.header('Réglage des paramètres')
        st.write("Scores obtenu :", get_score(model))
        
    if model == models[3]:
        best_params = load('best_params_lass.joblib')
        st.write('Meilleurs hyperparamètres pour le modèle:', best_params)
        st.header('Réglage des paramètres')
        st.write("Scores obtenu :", get_score(model))
        
    if model == models[4]:
        best_params = load('best_params_line.joblib')
        st.write('Meilleurs hyperparamètres pour le modèle:', best_params)
        st.header('Réglage des paramètres')
        st.write("Scores obtenu :", get_score(model))
        
    if model == models[5]:
        best_params = load('best_params_lasso_cv.joblib')
        st.write('Meilleurs hyperparamètres pour le modèle:', best_params)
        st.header('Réglage des paramètres')
        st.write("Scores obtenu :", get_score(model))