import pandas as pd
df=pd.read_csv("izmirim-kart-ulasim-istatistikleri.csv",sep=";")

def load_data(filepath):
    df=pd.read_csv("izmirim-kart-ulasim-istatistikleri.csv")
    print("data loaded...")
    return df
def show_statistics():
    print("size of the data: ",df.shape)
    print("number of missing values: ",df.isnull().sum().sum())
    print("mode of the categorical attribute INSTITUTION: ",df["INSTITUTION"].mode()[0])
show_statistics()
