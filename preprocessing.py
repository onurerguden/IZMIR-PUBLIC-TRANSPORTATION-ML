import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df=pd.read_csv("izmirim-kart-ulasim-istatistikleri.csv",sep=";")

'''dropped_columns=["TEACHER","SIXTY_YEARS_OLD","PERSONNEL","FREE"]

new_df=df.drop(columns=dropped_columns)

new_df.to_csv("izmirim-kart-temiz.csv",sep=";",index=False)'''

new_df=df

def discretization_equal_frequency():
    cols = ["STUDENT", "FULL_FARE"]
    kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    X_binned = kbd.fit_transform(new_df[cols])
    new_df["STUDENT_BINNED"] = X_binned[:, 0].astype(int)
    new_df["FULL_FARE_BINNED"] = X_binned[:, 1].astype(int)
    print(new_df[["STUDENT", "STUDENT_BINNED", "FULL_FARE", "FULL_FARE_BINNED"]].head(10))


def discretization_equal_interval():
    cols = ["STUDENT", "FULL_FARE"]
    kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform")
    X_binned = kbd.fit_transform(new_df[cols])
    new_df["STUDENT_BINNED_UNI"] = X_binned[:, 0].astype(int)
    new_df["FULL_FARE_BINNED_UNI"] = X_binned[:, 1].astype(int)
    print(new_df[["STUDENT", "STUDENT_BINNED_UNI", "FULL_FARE", "FULL_FARE_BINNED_UNI"]].head(10))



def normalize_and_standardize():
    scaler_norm = MinMaxScaler()
    scaler_std = StandardScaler()
    new_df["FULL_FARE_NORM"] = scaler_norm.fit_transform(new_df[["FULL_FARE"]])
    new_df["STUDENT_STD"] = scaler_std.fit_transform(new_df[["STUDENT"]])
    print(new_df[["FULL_FARE", "FULL_FARE_NORM", "STUDENT", "STUDENT_STD"]].head(10))



def one_hot_encoding():
    one_hot = pd.get_dummies(new_df["INSTITUTION"], prefix="INST")
    df_encoded = pd.concat([new_df, one_hot], axis=1)
    print(df_encoded.head())



def apply_outlier_detection(df, column):
    # Sadece numeric kolonlarda çalış
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Skipped non-numeric column: {column}")
        return df

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: {len(outliers)} outliers detected")

    return df
