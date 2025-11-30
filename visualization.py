import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import unicodedata

def save_plot(fig, filename):
    # Normalize and clean file names
    target_folder = os.path.join("plots", "preprocessing")
    normalized = (
        filename.replace("İ", "I").replace("ı", "i")
        .replace("Ş", "S").replace("ş", "s")
        .replace("Ğ", "G").replace("ğ", "g")
        .replace("Ü", "U").replace("ü", "u")
        .replace("Ö", "O").replace("ö", "o")
        .replace("Ç", "C").replace("ç", "c")
    )
    normalized = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    path = os.path.join(target_folder,normalized)

    if not os.path.exists(path):
        fig.savefig(path, format="pdf", bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        print(f"Plot already exists: {path}")

def plot_boxPlots(df):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    plt.figure(figsize=(14, 8))

# Renk paleti
    palette = sb.color_palette("Set2", len(numeric_columns))

# Boxplot çiz
    sb.boxplot(data=df[numeric_columns], palette=palette)

# X label rotasyonu
    plt.xticks(rotation=45)

# Log-scale (çok önemli)
    plt.yscale("log")

    plt.gcf().canvas.manager.set_window_title("Box Plots Improved")

# Kaydet
    save_plot(plt.gcf(), "box_plots_improved.pdf")

    plt.show()

def plot_barCharts(df):
    counts = df["INSTITUTION"].value_counts().head(20)

    plt.figure(figsize=(12, 10))

    # Renk paleti: 20 kategori için ideal
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(counts)))

    # Yatay bar chart
    plt.barh(counts.index, counts.values, color=colors)

    # Değerleri barların üzerine yaz
    for index, value in enumerate(counts.values):
        plt.text(value + max(counts.values) * 0.01, index, str(value), va='center')

    plt.xlabel("Frequency")

    # En yüksek barın yukarıda olması için
    plt.gca().invert_yaxis()

    plt.gcf().canvas.manager.set_window_title("Bar Chart - Top 20 Institutions (Improved)")
    save_plot(plt.gcf(), "bar_chart_top20_improved.pdf")

    plt.show()


def plot_scatterPlots(df):
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)

    plt.figure(figsize=(12,8))

    plt.scatter(
        df["FULL_FARE"],
        df["STUDENT"],
        c=df["DATE"].dt.month,
        cmap="viridis",
        alpha=0.4,
        s=20
    )

    plt.xlabel("FULL_FARE")
    plt.ylabel("STUDENT")

    plt.colorbar(label="Month")

    plt.gcf().canvas.manager.set_window_title("Scatter Plot - FULL_FARE vs STUDENT (Improved)")
    save_plot(plt.gcf(), "scatter_fullfare_vs_student_improved.pdf")

    plt.show()


def plot_linePlots(df):
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)

    eshot = df[df["INSTITUTION"] == "Eshot"].copy()

    eshot.set_index("DATE", inplace=True)

    monthly_avg = eshot["STUDENT"].resample("M").mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg.index, monthly_avg.values, marker="o")
    plt.xlabel("Month")
    plt.ylabel("Average Student Count")
    plt.grid(True)
    save_plot(plt.gcf(), "monthly_student_average_count_timeline.pdf")

#plt.show()
def plot_correlationHeatmap(df):
    numeric_df = df.select_dtypes(include=['int64','float64'])
    corr = numeric_df.corr()
    plt.figure(figsize=(12,8))
    sb.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.gcf().canvas.manager.set_window_title("Correlation Matrix Heatmap")
    save_plot(plt.gcf(), "correlation_heatmap.pdf")
    #plt.show()

def plot_all_vehicle_user_trends(df):
    """
    Generates monthly usage line charts by user types for all transportation vehicles (INSTITUTION)
    in the Izmirim Kart data.
    """
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)
    df["MONTH"] = df["DATE"].dt.to_period("M")

    user_types = ["FULL_FARE", "STUDENT", "TEACHER", "SIXTY_YEARS_OLD",
                  "TICKET", "CHILD", "PERSONNEL", "FREE", "BANK CARD"]

    colors = {
        "FULL_FARE": "#1f77b4",
        "STUDENT": "#ff7f0e",
        "TEACHER": "#2ca02c",
        "SIXTY_YEARS_OLD": "#d62728",
        "TICKET": "#9467bd",
        "CHILD": "#8c564b",
        "PERSONNEL": "#e377c2",
        "FREE": "#7f7f7f",
        "BANK CARD": "#00FFFF"  # light cyan
    }

    for inst in df["INSTITUTION"].unique():
        vdf = df[df["INSTITUTION"] == inst]
        monthly = vdf.groupby("MONTH")[user_types].sum()

        if monthly.empty:
            continue

        x_labels = [f"{d.month:02d}.{str(d.year)[2:]}" for d in monthly.index]

        plt.figure(figsize=(12,6))
        for col in user_types:
            plt.plot(x_labels, monthly[col], marker='o', label=col, color=colors.get(col, None))

        plt.xlabel("Month")
        plt.ylabel("Usage Count")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.gcf().canvas.manager.set_window_title(f"{inst} - Monthly Usage by User Types")
        filename = f"{inst.lower().replace(' ', '_')}_monthly_usage_by_user_types.pdf"
        save_plot(plt.gcf(), filename)
        #plt.show()

def plot_PCA_in_2D(pca_df):
    # INSTITUTION'ı sayısal kategoriye dönüştür
    colors = pca_df["INSTITUTION"].astype("category").cat.codes

    plt.figure(figsize=(10,8))
    plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=colors,
        cmap="tab20",
        alpha=0.6,
        s=30
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="INSTITUTION")

    save_plot(plt.gcf(), "pca_2d_visualization_improved.pdf")
    plt.show()


def plot_PCA_in_3D(pca_df,color="blue"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # INSTITUTION renklendirme
    colors = pca_df["INSTITUTION"].astype("category").cat.codes

    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        pca_df["PC3"],
        c=colors,
        cmap="tab20",
        s=40,
        alpha=0.6
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("INSTITUTION")

    save_plot(plt.gcf(), "pca_3d_visualization_improved.pdf")
    plt.show()


def show_all_plots(df):
    plot_linePlots(df)
    plot_boxPlots(df)
    plot_barCharts(df)
    plot_scatterPlots(df)
    plot_correlationHeatmap(df)
    plot_all_vehicle_user_trends(df)
