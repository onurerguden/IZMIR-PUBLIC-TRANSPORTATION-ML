from data_loader import *
from visualization import *
from preprocessing import *

def main():
    # 1. Veri yükleme ve temel görseller
    df = load_data()
    show_statistics(df)
    show_all_plots(df)  # Box, bar, scatter, line, correlation, trend

    # 2. Aykırı değer tespiti
    if "MONTH" not in df.columns:
        df["MONTH"] = pd.to_datetime(df["DATE"],dayfirst=True).dt.month

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print("Numeric columns to process:", numeric_cols)
    for column in numeric_cols:
        df = apply_outlier_detection(df, column)

    # 3. Discretization (Equal-frequency ve Equal-interval)
    print("\n--- Discretization (Equal-Frequency) ---")
    discretization_equal_frequency(df)

    print("\n--- Discretization (Equal-Interval) ---")
    discretization_equal_interval(df)

    # 4. Normalization & Standardization
    print("\n--- Normalization & Standardization ---")
    normalize_and_standardize(df)

    # 5. One-hot Encoding
    print("\n--- One-hot Encoding ---")
    one_hot_encoding(df)

    # 6. PCA (2D ve 3D Görselleştirme)
    print("\n--- PCA (2D) ---")
    pca_df_2d, variance_2d = apply_PCA(df, 2)
    pca_df_2d["INSTITUTION"] = df["INSTITUTION"].values
    plot_PCA_in_2D(pca_df_2d)

    print("\n--- PCA (3D) ---")
    pca_df_3d, variance_3d = apply_PCA(df, 3)
    pca_df_3d["INSTITUTION"] = df["INSTITUTION"].values
    plot_PCA_in_3D(pca_df_3d, "teal")

if __name__ == "__main__":
    main()
