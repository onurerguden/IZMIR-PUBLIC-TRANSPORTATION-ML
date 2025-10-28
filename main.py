from data_loader import *
from visualization import *

def main():
    df=load_data("izmirim-kart-ulasim-istatistikleri.csv")
    show_statistics(df)
    plot_boxPlots(df)
    plot_linePlots(df)
    plot_scatterPlots(df)
    plot_barCharts(df)
if __name__ == "__main__":
    main()