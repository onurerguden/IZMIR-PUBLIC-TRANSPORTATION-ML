from data_loader import *
from visualization import *

def main():
    df=load_data("izmirim-kart-ulasim-istatistikleri.csv")
    show_statistics(df)

    show_all_plots(df)

if __name__ == "__main__":
    main()