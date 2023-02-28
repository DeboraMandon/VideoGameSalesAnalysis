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
print(platform_count)

sns.set_theme(style="darkgrid")
g=sns.catplot(data=platform_count, kind="bar",x="Platform", y="Count", palette="dark", alpha=.6, height=10)
g.despine(left=True)
g.set_axis_labels("Platform", "Count")
g.tick_params(axis='x', rotation=70)
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
g.tick_params(axis='x', rotation=70)
plt.title("Répartition des Genres");