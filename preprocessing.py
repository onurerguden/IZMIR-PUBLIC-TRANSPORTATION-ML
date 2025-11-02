import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from data_loader import load_data


def discretization_equal_frequency(df):
    cols = ["STUDENT", "FULL_FARE"]
    kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    X_binned = kbd.fit_transform(df[cols])
    df["STUDENT_BINNED"] = X_binned[:, 0].astype(int)
    df["FULL_FARE_BINNED"] = X_binned[:, 1].astype(int)
    print(df[["STUDENT", "STUDENT_BINNED", "FULL_FARE", "FULL_FARE_BINNED"]].head(10))

def discretization_equal_interval(df):
    cols = ["STUDENT", "FULL_FARE"]
    kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform")
    X_binned = kbd.fit_transform(df[cols])
    df["STUDENT_BINNED_UNI"] = X_binned[:, 0].astype(int)
    df["FULL_FARE_BINNED_UNI"] = X_binned[:, 1].astype(int)
    print(df[["STUDENT", "STUDENT_BINNED_UNI", "FULL_FARE", "FULL_FARE_BINNED_UNI"]].head(10))



def normalize_and_standardize(df):
    scaler_norm = MinMaxScaler()
    scaler_std = StandardScaler()
    df["FULL_FARE_NORM"] = scaler_norm.fit_transform(df[["FULL_FARE"]])
    df["STUDENT_STD"] = scaler_std.fit_transform(df[["STUDENT"]])
    print(df[["FULL_FARE", "FULL_FARE_NORM", "STUDENT", "STUDENT_STD"]].head(10))



def one_hot_encoding(df):
    one_hot = pd.get_dummies(df["INSTITUTION"], prefix="INST")
    df_encoded = pd.concat([df, one_hot], axis=1)
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
