import pandas as pd
# Dosya adını değiştirdiğimiz için import da değişti
from dt_based_regressions import train_model, create_time_series_comparison_plots


def run_test():
    DOSYA_YOLU = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"

    # PARAMETRELER (Manuel Tarih Ayarı Buradan Yapılır)
    SECILEN_KURUM = "Hepsi"
    SECILEN_KART = "Hepsi"
    SECILEN_MODEL = "RandomForest"
    # Gelecek tahmini veya filtreleme için tarih buraya girilir:
    BASLANGIC_TARIHI = ""

    print("\n" + "=" * 60)
    print(f" TAHMIN MOTORU CALISTIRILIYOR")
    print(f" Hedef: {SECILEN_KURUM} - {SECILEN_KART}")
    print("=" * 60)

    try:
        print(f" Veri okunuyor: {DOSYA_YOLU} ...")
        df = pd.read_csv(DOSYA_YOLU, sep=";")

        print(f" Veri yuklendi. Satir sayisi: {len(df)}")
        print("-" * 30)

        model = train_model(
            df,
            target_col=SECILEN_KART,
            institution_name=SECILEN_KURUM,
            model_type=SECILEN_MODEL,
            start_date=BASLANGIC_TARIHI
        )


        print("\n" + "=" * 60)
        print(" ISLEM TAMAMLANDI!")
        print(f" Olusturulan grafikler 'plots' klasorune kaydedildi.")
        print("=" * 60)

        create_time_series_comparison_plots(df, output_dir="plots")

    except FileNotFoundError:
        print(f"\n HATA: '{DOSYA_YOLU}' bulunamadi!")
    except Exception as e:
        print(f"\n HATA OLUSTU:\n{e}")

if __name__ == "__main__":
    run_test()
