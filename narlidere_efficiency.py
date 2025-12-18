import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_percentage_error

# !!! BURASI KRİTİK: dt_based_regressions dosyasından import ediyoruz !!!
try:
    from regressions_for_findings import train_model
except ImportError:
    print("HATA: 'regressions_for_findings.py' dosyası bulunamadı!")


class Findings:
    def __init__(self, data_path):
        # CSV OKUMA (Noktalı virgül ayracı)
        self.df = pd.read_csv(data_path, sep=";", encoding="utf-8")
        self.df["DATE"] = pd.to_datetime(self.df["DATE"], format='mixed', dayfirst=True)

        # Çıktı klasörü
        self.output_dir = "plots/findings"
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_style("whitegrid")

    def _train_and_predict(self, institution, t_start, t_end, test_start, test_end):
        """
        dt_based_regressions modülünü kullanarak eğitim ve tahmin yapar.
        """
        print(f">>> {institution} için DT Based Model çalışıyor...")
        print(f"    Eğitim: {t_start} -> {t_end}")
        print(f"    Tahmin: {test_start} -> {test_end}")

        # dt_based_regressions.train_model fonksiyonunu çağırıyoruz
        model, results_df = train_model(
            df=self.df,  # Ham veriyi gönderiyoruz
            target_col="Hepsi",
            institution_name=institution,
            model_type="XGBoost",
            train_start=t_start,
            train_end=t_end,
            test_start=test_start,
            test_end=test_end
        )

        if results_df is None or results_df.empty:
            print(f"HATA: {institution} için sonuç dönmedi.")
            return None

        return results_df[['DATE', 'TOTAL_PASSENGER', 'PREDICTED']]

    # =========================================================================
    # ANALİZ 1: NARLIDERE METROSU AÇILIŞ ETKİSİ (GENEL)
    # =========================================================================
    def analyze_metro_domination(self):
        TRAIN_START = "2021-01-01"
        TRAIN_END = "2024-02-23"
        TEST_START = "2024-04-15"
        TEST_END = "2024-12-10"

        print(f"\n=== ANALİZ 1: GENEL AÇILIŞ ETKİSİ ===")
        eshot = self._train_and_predict("ESHOT", TRAIN_START, TRAIN_END, TEST_START, TEST_END)
        metro = self._train_and_predict("METRO", TRAIN_START, TRAIN_END, TEST_START, TEST_END)

        if eshot is not None and metro is not None:
            self._calculate_and_plot(eshot, metro, "metro_domination_impact",
                                     "Narlıdere Metrosu Sonrası Dominasyon Değişimi")

    # =========================================================================
    # ANALİZ 2: TOPLAM ETKİ (AÇILIŞ + 8 DK SIKLIK)
    # =========================================================================
    def analyze_frequency_impact(self):
        TRAIN_START = "2021-01-01"
        TRAIN_END = "2024-02-23"  # Eski sistem (Hat yok)
        TEST_START = "2024-12-11"  # Yeni sistem (8 dk sıklık)
        TEST_END = "2025-04-21"  # Güncel son tarih

        print(f"\n=== ANALİZ 2: TOPLAM ETKİ (HAT AÇILIŞI + 8 DK SIKLIK) ===")
        eshot = self._train_and_predict("ESHOT", TRAIN_START, TRAIN_END, TEST_START, TEST_END)
        metro = self._train_and_predict("METRO", TRAIN_START, TRAIN_END, TEST_START, TEST_END)

        if eshot is not None and metro is not None:
            self._calculate_and_plot(eshot, metro, "frequency_total_impact",
                                     "Eski Sisteme Kıyasla Yeni Dönem (Hat Açılışı + 8 Dk) Etkisi")

    # =========================================================================
    # ORTAK HESAPLAMA VE ÇİZİM (GÜNCELLENDİ)
    # =========================================================================
    def _calculate_and_plot(self, eshot_df, metro_df, filename_prefix, plot_title):
        merged = pd.merge(eshot_df, metro_df, on="DATE", suffixes=('_ESHOT', '_METRO'))

        # 1. Oransal Hesaplama (Dominasyon)
        merged['ACTUAL_RATIO'] = merged['TOTAL_PASSENGER_METRO'] / merged['TOTAL_PASSENGER_ESHOT']
        merged['PREDICTED_RATIO'] = merged['PREDICTED_METRO'] / merged['PREDICTED_ESHOT']

        avg_act_ratio = merged['ACTUAL_RATIO'].mean()
        avg_pred_ratio = merged['PREDICTED_RATIO'].mean()
        change_pct = ((avg_act_ratio - avg_pred_ratio) / avg_pred_ratio) * 100

        # 2. Sayısal Hesaplama (Günlük Yolcu Kazanımı) - YENİ KISIM
        # Metro için: Gerçekleşen - Modelin Tahmin Ettiği (Beklenen)
        merged['METRO_UPLIFT'] = merged['TOTAL_PASSENGER_METRO'] - merged['PREDICTED_METRO']
        avg_uplift = merged['METRO_UPLIFT'].mean()
        total_uplift = merged['METRO_UPLIFT'].sum()

        print(f"\n--- SONUÇLAR: {plot_title} ---")
        print(f"🔹 ORANSAL DEĞİŞİM (PAZAR PAYI):")
        print(f"   Model Beklentisi (Eski Trend): {avg_pred_ratio:.4f}")
        print(f"   Gerçekleşen (Yeni Durum)     : {avg_act_ratio:.4f}")
        print(f"   FARK (Dominasyon Artışı)     : %{change_pct:.2f}")

        print(f"🔹 SAYISAL KAZANIM (METRO):")
        print(f"   Günlük Ortalama Ekstra Yolcu : +{int(avg_uplift):,} Kişi")
        print(f"   Toplam Kazanılan Yolcu       : +{int(total_uplift):,} Kişi (Analiz Süresince)")

        # 3. Çizim (Oran Grafiği - Mevcut Halini Koruyoruz)
        plt.figure(figsize=(14, 7))
        plt.plot(merged['DATE'], merged['ACTUAL_RATIO'], color='#2ca02c', linewidth=2.5, label='Gerçekleşen')
        plt.plot(merged['DATE'], merged['PREDICTED_RATIO'], color='#d62728', linestyle='--', linewidth=2, alpha=0.8,
                 label='Model Beklentisi (Eski Trend)')

        plt.axhline(avg_act_ratio, color='#2ca02c', linestyle=':', alpha=0.5)
        plt.axhline(avg_pred_ratio, color='#d62728', linestyle=':', alpha=0.5)

        plt.fill_between(merged['DATE'], merged['PREDICTED_RATIO'], merged['ACTUAL_RATIO'],
                         where=(merged['ACTUAL_RATIO'] > merged['PREDICTED_RATIO']),
                         color='green', alpha=0.15, label='Metro Kazanımı (Pozitif Etki)')

        # Grafiğin üzerine sayısal kazanımı da not düşelim
        plt.title(f"{plot_title}\n(Günlük Ort. Kazanım: +{int(avg_uplift):,} Yolcu)", fontsize=14)
        plt.ylabel('Metro / ESHOT Oranı')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/{filename_prefix}.png")
        plt.close()


if __name__ == "__main__":
    findings = Findings("izmirim-kart-ulasim-istatistikleri-guncel-extended.csv")
    findings.analyze_metro_domination()
    findings.analyze_frequency_impact()