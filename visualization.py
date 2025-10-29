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
    plt.figure(figsize=(12,8))
    sb.boxplot(data=df[numeric_columns])
    plt.title('Box Plots')
    plt.xticks(rotation=45)
    plt.gcf().canvas.manager.set_window_title("Box Plots")
    plt.show()
def plot_barCharts(df):
    counts=df["INSTITUTION"].value_counts().head(20)
    plt.figure(figsize=(12,8))
    counts.plot(kind='bar')
    plt.title("Top 20 Institutions")
    plt.ylabel("FREQUENCY")
    plt.xlabel("INSTITUTION")
    plt.gcf().canvas.manager.set_window_title("Bar Chart - Top 20 Institutions")
    plt.show()
def plot_scatterPlots(df):
    plt.figure(figsize=(12,8))
    plt.scatter(df["FULL_FARE"],df["STUDENT"])
    plt.title("FULL_FARE vs STUDENT")
    plt.xlabel("FULL_FARE")
    plt.ylabel("STUDENT")
    plt.gcf().canvas.manager.set_window_title("Scatter Plot - FULL_FARE vs STUDENT")
    plt.show()
def plot_linePlots(df):
    df["DATE"]=pd.to_datetime(df["DATE"],dayfirst=True)
    eshot=df[df["INSTITUTION"]=="Eshot"]
    plt.figure(figsize=(12,8))
    plt.plot(eshot["DATE"],eshot["STUDENT"])
    plt.title("Eshot - Student Usage Over Time")
    plt.xlabel("Date")
    plt.ylabel("STUDENT")
    plt.gcf().canvas.manager.set_window_title("Line Plot - Eshot Student Usage Over Time")
    plt.show()
