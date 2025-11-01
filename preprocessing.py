import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df=pd.read_csv("izmirim-kart-ulasim-istatistikleri.csv",sep=";")

'''dropped_columns=["TEACHER","SIXTY_YEARS_OLD","PERSONNEL","FREE"]

new_df=df.drop(columns=dropped_columns)

new_df.to_csv("izmirim-kart-temiz.csv",sep=";",index=False)'''

new_df=df


### GOZDEN GECIRILECEK -OE
def discretization_equal_frequency():
    X=new_df[["STUDENT","FULL_FARE"]]
    kbd=KBinsDiscretizer(n_bins=4,encode="ordinal",strategy="quantile")
    X_binned=kbd.fit_transform(X)
    new_df["FULL_FARE_BINNED"]=X_binned[:,1]
    new_df["STUDENT_BINNED"]=X_binned[:,0]
    print(new_df[["FULL_FARE", "FULL_FARE_BINNED", "STUDENT", "STUDENT_BINNED"]].head(10))


### GOZDEN GECIRILECEK -OE
def discretization_equal_interval():
    X=new_df[["STUDENT","FULL_FARE"]]
    kbd=KBinsDiscretizer(n_bins=4,encode="ordinal",strategy="uniform",subsample=200_000)
    X_binned=kbd.fit_transform(X)
    new_df["FULL_FARE_BINNED"]=X_binned[:,1]
    new_df["STUDENT_BINNED"]=X_binned[:,0]
    print("\n",new_df[["FULL_FARE", "FULL_FARE_BINNED", "STUDENT", "STUDENT_BINNED"]].head(1000))




### GOZDEN GECIRILECEK -OE
def normalize_and_standardize():
    scaler_norm=MinMaxScaler()
    scaler_std=StandardScaler()
    new_df["FULL_FORM_NORM"]=scaler_norm.fit_transform(new_df[["FULL_FARE"]])#normalization on full fare
    new_df["STUDENT_STD"]=scaler_std.fit_transform(new_df[["STUDENT"]])#standardization on student

def one_hot_encoding():
    one_hot=pd.get_dummies(new_df["INSTITUTION"],prefix="INST")
    df_encoded=pd.concat([new_df,one_hot],axis=1)
    '''pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)'''
    print(df_encoded.head())
def apply_outlier_detection(column):
    Q1=new_df[column].quantile(0.25)
    Q3=new_df[column].quantile(0.75)
    IQR=Q3-Q1
    lowerBound=Q1-1.5*IQR
    upperBound=Q3+1.5*IQR

    outliers=new_df[(df[column]<lowerBound)|(df[column]>upperBound)]
    print(column,": ",outliers.shape[0])

