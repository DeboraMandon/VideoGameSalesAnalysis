import pandas as pd

path="https://raw.githubusercontent.com/DeboraMandon/video_game_sales_analysis/main/vgsales.csv"
df=pd.read_csv(path, index_col=0)
display(df.head())
print(df.shape)

# TYPE DE VARIABLE

display(df.info())

# TAUX DE NA

display(df.isna().sum())
#pourcentage de valeur nulle pour la variable Year
Year_Na=(df['Year'].isna().sum())/(len(df['Year']))
Year_Na=Year_Na*100
print('Year',round(Year_Na,2),'%')

#pourcentage de valeur nulle pour la variable Publisher
Year_Na=(df['Publisher'].isna().sum())/(len(df['Publisher']))
Year_Na=Year_Na*100
print('Publisher',round(Year_Na,2),'%')

# DISTRIBUTION DES VALEURS

display(df.describe())
display(df.describe(include='O'))

# ETENDUE DES VALEURS

display(df['Platform'].unique())
display(df['Year'].unique())
display(df['Genre'].unique())

