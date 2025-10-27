import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("izmirim-kart-ulasim-istatistikleri.csv",sep=";")
#print(df.head())
print(df.columns)
print(df.shape)
print(df.isnull().sum())
nullRows=df[df.isnull().any(axis=1)]
print(nullRows)
'''
print(df.columns)
print(df.info())
print(df.describe())'''
def printColumns():
    with open("izmirim-kart-ulasim-istatistikleri.csv") as f:
        print(f.readline())
#printColumns()
def drawCorrMatrix():
    corrMatrix=df.corr(numeric_only=True)
    print(corrMatrix)
    sns.heatmap(corrMatrix,annot=True,cmap="coolwarm")
    plt.show()
'''plt.scatter(df["FULL_FARE"],df["SIXTY_YEARS_OLD"])
plt.xlabel("fullfare")
plt.ylabel("SIXTY")
plt.show()'''
sns.scatterplot(data=df,x="FULL_FARE",y="SIXTY_YEARS_OLD")
plt.show()




