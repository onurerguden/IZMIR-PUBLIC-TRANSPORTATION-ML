from data_loader import *
from visualization import *
from preprocessing import *

def main():
    df=load_data("izmirim-kart-ulasim-istatistikleri.csv")
    show_statistics(df)
    #show_all_plots(df)
   # apply_outlier_detection("TICKET")
    print(df.columns)
    for column in df.columns:
        if column!="DATE" and column!="INSTITUTION":
            apply_outlier_detection(column)
if __name__ == "__main__":
    main()
