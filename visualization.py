import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df=pd.read_csv("izmirim-kart-ulasim-istatistikleri.csv",sep=";")
def plot_boxPlots(df):
    numeric_columns=df.select_dtypes(include=['int64','float64']).columns
    '''for column in numeric_columns:
        plt.figure(figsize=(6,4))
        sb.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        plt.xlabel(column)
        plt.show()'''
    plt.figure(figsize=(10,6))
    sb.boxplot(data=df[numeric_columns])
    plt.title('Box Plots')
    plt.xticks(rotation=45)
    plt.show()
#plot_boxPlots(df)
def plot_barCharts(df):
    counts=df["INSTITUTION"].value_counts().head(20)
    counts.plot(kind='bar')
    plt.title("Top 20 Institutions")
    plt.ylabel("FREQUENCY")
    plt.xlabel("INSTITUTION")
    plt.show()
#plot_barCharts(df)
def plot_scatterPlots(df):
    plt.scatter(df["FULL_FARE"],df["STUDENT"])
    plt.title("FULL_FARE vs STUDENT")
    plt.xlabel("FULL_FARE")
    plt.ylabel("STUDENT")
    plt.show()
#plot_scatterPlots(df)
def plot_linePlots(df):
    df["DATE"]=pd.to_datetime(df["DATE"],dayfirst=True)
    eshot=df[df["INSTITUTION"]=="Eshot"]
    plt.plot(eshot["DATE"],eshot["STUDENT"])
    plt.title("Eshot - Student Usage Over Time")
    plt.xlabel("Date")
    plt.ylabel("STUDENT")
    plt.show()
plot_linePlots(df)