import pandas as pd
from dt_based_regressions import train_ultimate_model


def run_ultimate_test():
    # CSV Dosyanızın Tam Adı
    DOSYA_YOLU = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"

    # PARAMETRELER
    SECILEN_KURUM = "Tramvay Konak"  # Örn: Eshot, Metro, Hepsi
    SECILEN_KART = "BANK CARD"  # Örn: STUDENT, FULL_FARE, Hepsi
    SECILEN_MODEL = "Random Forest"  # Grafiklerde hepsi çıkacak ama detay analizi bunun için yapılacak

    print("\n" + "=" * 60)
    print(f" ULTIMATE TAHMİN MOTORU ÇALIŞTIRILIYOR")
    print(f" Hedef: {SECILEN_KURUM} - {SECILEN_KART}")
    print("=" * 60)

    try:
        print(f" Veri okunuyor: {DOSYA_YOLU} ...")
        # Noktalı virgül (;) ile ayrılmışsa sep=";" kullanın
        df = pd.read_csv(DOSYA_YOLU, sep=";")

        print(f" Veri yüklendi. Satır sayısı: {len(df)}")
        print("-" * 30)

        # Modeli eğit ve grafikleri oluştur
        model = train_ultimate_model(
            df,
            target_col=SECILEN_KART,
            institution_name=SECILEN_KURUM,
            model_type=SECILEN_MODEL,
            start_date="2023-08-21"
        )

        print("\n" + "=" * 60)
        print(" İŞLEM TAMAMLANDI!")
        print(f" Şu grafikleri 'plots' klasöründe bulabilirsin:")
        print(" 1. Residuals_vs_Predicted_XGB.png (Kırmızı - İstenilen Grafik)")
        print(" 2. Residuals_vs_Predicted_RF.png (Mavi - İstenilen Grafik)")
        print(" 3. Residuals_vs_Predicted_Ridge.png (Gri - İstenilen Grafik)")
        print(" 4. 00_model_comparison.png (Bar Chart)")
        print("=" * 60)

    except FileNotFoundError:
        print(f"\n HATA: '{DOSYA_YOLU}' bulunamadı!")
    except Exception as e:
        print(f"\n HATA OLUŞTU:\n{e}")


if __name__ == "__main__":
    run_ultimate_test()