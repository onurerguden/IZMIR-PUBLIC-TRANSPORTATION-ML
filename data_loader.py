import pandas as pd

def load_data():
    df=pd.read_csv("izmirim-kart-ulasim-istatistikleri-guncel.csv",sep=",")
    df.drop("_id",axis=1,inplace=True)
    print("data loaded...")
    return df
def show_statistics(df):
    print("size of the data: ",df.shape)
    print("number of missing values: ",df.isnull().sum().sum())
    print("mode of the categorical attribute INSTITUTION: ",df["INSTITUTION"].mode()[0])


