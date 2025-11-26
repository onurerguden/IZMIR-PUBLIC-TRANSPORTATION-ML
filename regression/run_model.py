import pandas as pd
# Algoritmanızı regressiontwo dosyasından çekiyoruz
from regressiontwo import train_monthly_xgboost_student

def run_test_scenario():
    print("--------------------------------------------------")
    print("🚀 ESHOT ÖĞRENCİ TAHMİN ALGORİTMASI BAŞLATILIYOR")
    print("--------------------------------------------------")

    # 1. Veriyi Yükle
    # 'group_dataset_by_TimeLines.py' çıktısı olan dosyayı kullanıyoruz
    file_path = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"

    try:
        # Dosyanız noktalı virgül (;) ile ayrılmıştı, bunu belirtiyoruz
        df = pd.read_csv(file_path, sep=";")
        print(f"✅ Veri Seti Yüklendi: {file_path}")
        print(f"📊 Toplam Satır Sayısı: {len(df)}")

    except FileNotFoundError:
        print(f"❌ HATA: '{file_path}' dosyası bulunamadı!")
        print("Lütfen önce veri setini oluşturduğunuzdan emin olun.")
        return

    # 2. Algoritmayı Çalıştır
    # Bu fonksiyon kendi içinde temizlik, %80-%20 ayrımı ve test işlemini yapar
    try:
        print("\n⏳ Model eğitiliyor ve test ediliyor, lütfen bekleyin...\n")

        models, results_table = train_monthly_xgboost_student(df, target_col="STUDENT")

        # 3. Sonuçları Yazdır
        print("--------------------------------------------------")
        print("🏆 MODEL PERFORMANS SONUÇLARI (TEST VERİSİ)")
        print("--------------------------------------------------")
        # Tabloyu düzgün formatta yazdıralım
        print(results_table.to_string(index=False))
        print("--------------------------------------------------")
        print("✅ İşlem Başarıyla Tamamlandı.")

    except Exception as e:
        print(f"\n❌ Algoritma çalışırken bir hata oluştu:\n{e}")

if __name__ == "__main__":
    run_test_scenario()