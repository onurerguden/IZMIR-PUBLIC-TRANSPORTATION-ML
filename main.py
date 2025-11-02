from data_loader import *
from visualization import *
from preprocessing import *

def main():
    df=load_data()
    show_statistics(df)
    show_all_plots(df)

    if "MONTH" not in df.columns:
        df["MONTH"] = pd.to_datetime(df["DATE"]).dt.month

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print("Numeric columns to process:", numeric_cols)

    for column in numeric_cols:
        df = apply_outlier_detection(df, column)

    pca_df,variance=apply_PCA(df,3)
    plot_PCA_in_3D(pca_df,"teal")

if __name__ == "__main__":
    main()
