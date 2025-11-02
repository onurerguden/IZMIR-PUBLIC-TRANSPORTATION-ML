import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from data_loader import load_data


def plot_boxPlots(df):
    numeric_columns = df.select_dtypes(include=['int64','float64']).columns
    plt.figure(figsize=(12,8))
    sb.boxplot(data=df[numeric_columns])
    plt.title('Box Plots')
    plt.xticks(rotation=45)
    plt.gcf().canvas.manager.set_window_title("Box Plots")
    plt.show()

def plot_barCharts(df):
    counts = df["INSTITUTION"].value_counts().head(20)
    plt.figure(figsize=(12,8))
    counts.plot(kind='bar')
    plt.title("Top 20 Institutions")
    plt.ylabel("FREQUENCY")
    plt.xlabel("INSTITUTION")
    plt.gcf().canvas.manager.set_window_title("Bar Chart - Top 20 Institutions")
    plt.show()

def plot_scatterPlots(df):
    plt.figure(figsize=(12,8))
    plt.scatter(df["FULL_FARE"], df["STUDENT"])
    plt.title("FULL_FARE vs STUDENT")
    plt.xlabel("FULL_FARE")
    plt.ylabel("STUDENT")
    plt.gcf().canvas.manager.set_window_title("Scatter Plot - FULL_FARE vs STUDENT")
    plt.show()

def plot_linePlots(df):
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)
    eshot = df[df["INSTITUTION"] == "Eshot"]
    plt.figure(figsize=(12,8))
    plt.plot(eshot["DATE"], eshot["STUDENT"])
    plt.title("Eshot - Student Usage Over Time")
    plt.xlabel("Date")
    plt.ylabel("STUDENT")
    plt.gcf().canvas.manager.set_window_title("Line Plot - Eshot Student Usage Over Time")
    plt.show()

def plot_correlationHeatmap(df):
    numeric_df = df.select_dtypes(include=['int64','float64'])
    corr = numeric_df.corr()
    plt.figure(figsize=(12,8))
    sb.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.gcf().canvas.manager.set_window_title("Correlation Matrix Heatmap")
    plt.show()

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

        plt.title(f"{inst} - Monthly Usage by User Types")
        plt.xlabel("Month")
        plt.ylabel("Usage Count")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.gcf().canvas.manager.set_window_title(f"{inst} - Monthly Usage by User Types")
        plt.show()

def show_all_plots(df):
    plot_boxPlots(df)
    plot_barCharts(df)
    plot_scatterPlots(df)
    plot_linePlots(df)
    plot_correlationHeatmap(df)
    plot_all_vehicle_user_trends(df)
