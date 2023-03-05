import pandas as pd

path="https://raw.githubusercontent.com/DeboraMandon/video_game_sales_analysis/main/vgsales.csv"
df=pd.read_csv(path, index_col=0)
print(df.head())
print(df.shape)

# TYPE DE VARIABLE

print(df.info())

# TAUX DE NA

print(df.isna().sum())
#pourcentage de valeur nulle pour la variable Year
Year_Na=(df['Year'].isna().sum())/(len(df['Year']))
Year_Na=Year_Na*100
print('Year',round(Year_Na,2),'%')

#pourcentage de valeur nulle pour la variable Publisher
Year_Na=(df['Publisher'].isna().sum())/(len(df['Publisher']))
Year_Na=Year_Na*100
print('Publisher',round(Year_Na,2),'%')

# DISTRIBUTION DES VALEURS

print(df.describe())
print(df.describe(include='O'))

# ETENDUE DES VALEURS

print(df['Platform'].unique())
print(df['Year'].unique())
print(df['Genre'].unique())


# RELATION ENTRE LES VALEURS
########## PART D
import matplotlib.pyplot as plt
import seaborn as sns


#PLATEFORMES

platform_count=df.groupby('Platform').agg('count')
platform_count.reset_index(inplace=True)
platform_count.drop(['Year', 'Genre','Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'], axis=1, inplace=True)
platform_count=platform_count.rename(columns={'Platform':'Platform','Name':'Count'})

sns.set_theme(style="darkgrid")
g=sns.catplot(data=platform_count, kind="bar",x="Platform", y="Count", palette="dark", alpha=.6, height=10)
g.despine(left=True)
g.set_axis_labels("Platform", "Count")
g.set_xticklabels(rotation=70)
plt.title("Répartition des plateformes");



#GENRES

genre_count=df.groupby('Genre').agg('count')
genre_count.reset_index(inplace=True)
genre_count.drop(['Year', 'Platform','Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'], axis=1, inplace=True)
genre_count=genre_count.rename(columns={'Genre':'Genre','Name':'Count'})
print(genre_count)

sns.set_theme(style="darkgrid")
g=sns.catplot(data=genre_count, kind="bar",x="Genre", y="Count", palette="dark", alpha=.6, height=10)
g.despine(left=True)
g.set_axis_labels("Genre", "Count")
g.set_xticklabels(rotation=70)
plt.title("Répartition des Genres");


plt.figure(figsize=(20,8))
plt.title('Effectif de chaque Genre de ventes')

sns.countplot(x = 'Genre', hue = 'cat_Global_Sales', data = df);


# AGGREGATION POUR ETUDE DES VENTES EN FONCTION DU GENRE

agg_genre=df.groupby(['Genre'], as_index=False).agg({'Global_Sales':'sum', 'NA_Sales':'sum', 'EU_Sales':'sum', 'JP_Sales':'sum', 'Other_Sales':'sum'})
agg_genre=agg_genre.sort_values(by='Genre', ascending=True)

plt.figure()
plt.title('Ventes par Genre', fontsize=18)
sns.barplot(x='Genre', y='Global_Sales', order=agg_genre['Genre'].values, data=agg_genre);
plt.xticks(rotation=70);

vente_best=0
for i in agg_genre['Genre'].iloc[-1:]:
    sale=agg_genre[agg_genre['Genre']==i]['Global_Sales'].sum()
    vente_best += sale

vente_worst=0
for i in agg_genre['Genre'].iloc[:1]:
    sale=agg_genre[agg_genre['Genre']==i]['Global_Sales'].sum()
    vente_worst += sale

vente_best_genre=vente_best/agg_genre.Global_Sales.sum()*100
vente_worst_genre=vente_worst/agg_genre.Global_Sales.sum()*100


print('Pourcentage des ventes du Genre qui se vend le plus:',round(vente_best_genre,2),'%')
print('Pourcentage des ventes du Genre qui se vend le plus:',round(vente_worst_genre,2),'%')


fig,ax=plt.subplots(4,3,figsize=(45,25))

coordonnees = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2)]

for i, j in zip(coordonnees, df['Genre'].unique()):
    ax[i[0], i[1]].plot_date(x=df[df['Genre']==j]['Year'].values,
                             y=df[df['Genre']==j]['Global_Sales'].values,
                             xdate=True,
                             ls='-')
    ax[i[0], i[1]].set_title(str(j), fontsize=30);



# ETUDE DES VARIABLES NUMERIQUES

for i in var_num:
    df['cat_'+i]=pd.qcut(df[i], q=[0,.25,.5,.75,1.], duplicates='drop')

df['Year']=pd.to_datetime(df['Year'], format='%Y')


sales_per_year=df.groupby('Year', as_index=False).agg({'NA_Sales':sum, 'EU_Sales':sum, 'JP_Sales':sum, 'Other_Sales':sum,'Global_Sales':sum})


sns.set(style="whitegrid")
plt.figure(figsize=(15,6))
plt.plot_date(x=sales_per_year['Year'].values,
              y=sales_per_year['Global_Sales'].values,
              xdate=True,
              ls='-',
              label='Global Sales')
plt.plot_date(x=sales_per_year['Year'].values,
              y=sales_per_year['NA_Sales'].values,
              xdate=True,
              ls='-',
              label='North America Sales')
plt.plot_date(x=sales_per_year['Year'].values,
              y=sales_per_year['EU_Sales'].values,
              xdate=True,
              ls='-',
              label='Europe Sales')
plt.plot_date(x=sales_per_year['Year'].values,
              y=sales_per_year['JP_Sales'].values,
              xdate=True,
              ls='-',
              label='Japan Sales')
plt.plot_date(x=sales_per_year['Year'].values,
              y=sales_per_year['Other_Sales'].values,
              xdate=True,
              ls='-',
              label='Other Sales')
plt.legend(loc='best')
plt.title('Evolution des ventes')
plt.show();


#PLATEFORMES

plt.figure(figsize=(20,8))
plt.title('Effectif de chaque Plateformes de ventes')

sns.countplot(x = 'Platform', hue = 'cat_Global_Sales', data = df);


#GENRE

plt.figure(figsize=(20,8))
plt.title('Effectif de chaque Genre de ventes')

sns.countplot(x = 'Genre', hue = 'cat_Global_Sales', data = df);


# ANALYSE DE LA TENDANCE DES VENTES GLOBALES

# Calcul des ventes totales
total_sales = df.groupby('Year')['Global_Sales'].sum()

# Calcul des ventes totales cumulées
cumulative_total_sales = total_sales.cumsum()

# Affichage de la série temporelle
plt.plot_date(cumulative_total_sales.index, cumulative_total_sales.values);


plt.figure(figsize = (17,8))

plt.subplot(321)
sns.boxplot(x = 'NA_Sales', data = df);
plt.title('Boxplot des Ventes en Amérique du Nord')

plt.subplot(322)
sns.boxplot(x = 'EU_Sales', data = df)
plt.title('Boxplot des Ventes en Europe');

plt.subplot(323)
sns.boxplot(x = 'JP_Sales', data = df);
plt.title('Boxplot des Ventes au Japon')

plt.subplot(324)
sns.boxplot(x = 'Other_Sales', data = df)
plt.title('Boxplot des Ventes dans les Autres Pays');

plt.subplot(325)
sns.boxplot(x = 'Global_Sales', data = df);
plt.title('Boxplot des Ventes au Global');


# ANALYSE DES VARIABLES QUALITATIVES

Genre_count=pd.DataFrame(df['Genre'].value_counts())
Platform_count=pd.DataFrame(df['Platform'].value_counts())

plt.figure(figsize = (10,10))

plt.subplot(221)
sns.boxplot(y = 'Genre', data = Genre_count);
plt.title('Boxplot des Genres de Jeux Videos')

plt.subplot(222)
sns.boxplot(y = 'Platform', data = Platform_count)
plt.title('Boxplot des Platforms de Jeux Videos');


# TESTS DE CORRELATION ENTRE LE GLOBAL ET NA_SALES

from scipy.stats import pearsonr

pd.DataFrame(pearsonr(sales_per_year['Global_Sales'], 
                      sales_per_year['NA_Sales']), 
                      index = ['pearson_coeff', 'p-value'], 
                      columns = ['resultat_test'])

# TESTS DE CORRELATION ENTRE LE GLOBAL ET EU_SALES

from scipy.stats import pearsonr

pd.DataFrame(pearsonr(sales_per_year['Global_Sales'], 
                      sales_per_year['EU_Sales']), 
                      index = ['pearson_coeff', 'p-value'], 
                      columns = ['resultat_test'])

# TESTS DE CORRELATION ENTRE LE GLOBAL ET JP_SALES

from scipy.stats import pearsonr

pd.DataFrame(pearsonr(sales_per_year['Global_Sales'], 
                      sales_per_year['JP_Sales']), 
                      index = ['pearson_coeff', 'p-value'], 
                      columns = ['resultat_test'])

# TESTS DE CORRELATION ENTRE LE GLOBAL ET OTHER_SALES

from scipy.stats import pearsonr

pd.DataFrame(pearsonr(sales_per_year['Global_Sales'], 
                      sales_per_year['Other_Sales']), 
                      index = ['pearson_coeff', 'p-value'], 
                      columns = ['resultat_test'])


sns.jointplot('NA_Sales', 'Global_Sales', 
              data = sales_per_year, kind = 'reg');

sns.jointplot('EU_Sales', 'Global_Sales', 
              data = sales_per_year, kind = 'reg');

sns.jointplot('JP_Sales', 'Global_Sales', 
              data = sales_per_year, kind = 'reg');

sns.jointplot('Other_Sales', 'Global_Sales', 
              data = sales_per_year, kind = 'reg');



