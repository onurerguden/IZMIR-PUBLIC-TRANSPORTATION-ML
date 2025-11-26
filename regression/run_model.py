import pandas as pd
from regressiontwo import train_general_xgboost_model


def run_test_scenario():
    # ==========================================
    # KONTROL PANELI
    # ==========================================

    SECILEN_KURUM = "ÝZTAÞIT KÝRAZ"  # Orn: Eshot, Metro, Izban
    SECILEN_KART = "STUDENT"  # Orn: STUDENT, FULL_FARE

    # --- TARIH AYARI ---
    # Filtrelemek isterseniz tarih girin: "2023-07-01"
    # Filtreyi KAPATMAK (Tum veri) icin: None
    BASLANGIC_TARIHI = None

    # ==========================================

    print("--------------------------------------------------")
    print(f"TAHMIN ALGORITMASI: {SECILEN_KURUM.upper()} - {SECILEN_KART}")
    print(f"Tarih Modu: {BASLANGIC_TARIHI if BASLANGIC_TARIHI else 'TUM TARIHCE'}")
    print("--------------------------------------------------")

    file_path = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"

    try:
        df = pd.read_csv(file_path, sep=";")
        print(f"Veri Seti Yuklendi. Toplam Satir: {len(df)}")
    except FileNotFoundError:
        print(f"HATA: '{file_path}' bulunamadi!")
        return

    try:
        # Tarih parametresini gonderiyoruz
        model, results_table = train_general_xgboost_model(
            df,
            target_col=SECILEN_KART,
            institution_name=SECILEN_KURUM,
            start_date=BASLANGIC_TARIHI  # <--- YENI PARAMETRE
        )

        print("\n" + "=" * 80)
        print(f"SONUC RAPORU ({SECILEN_KURUM} - {SECILEN_KART})")
        print("=" * 80)
        print(results_table.to_string(index=False))
        print("=" * 80)

    except Exception as e:
        print(f"\nHata olustu:\n{e}")


if __name__ == "__main__":
    run_test_scenario()