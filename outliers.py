import pandas as pd
import numpy as np
import sys
from collections import Counter


class OutlierDetector:

    def __init__(self, df):
        """
        Veri setini yükler, tarih formatını düzeltir (SWAP işlemi) ve hazırlıkları yapar.
        """
        self.df = df.copy()

        # --- HAFIZA (Global Outlier Toplayici) ---
        self.all_outlier_dates = []

        # 1. Tarih Formatı Kontrolü
        if 'DATE' not in self.df.columns:
            print("HATA: CSV dosyasında 'DATE' sütunu bulunamadı!")
            sys.exit(1)

        # ---------------------------------------------------------
        # ADIM 1: TARİHLERİ OKU VE DÜZELT (CRITICAL FIX)
        # ---------------------------------------------------------
        print("🛠️ Tarih formatları düzeltiliyor (DD/MM Swap)...")

        # Önce standart çevirim yapıyoruz
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], dayfirst=True, errors='coerce')

        # Düzeltme Fonksiyonu: 2021, 2022, 2024 yıllarında Gün <= 12 ise Yer Değiştir
        def fix_date_anomaly(row):
            if pd.isnull(row): return row
            # Şüpheli Yıllar ve Şüpheli Günler (Ay olabilecek günler)
            if row.year in [2021, 2022, 2024] and row.day <= 12:
                try:
                    # Günü Ay yap, Ayı Gün yap
                    return pd.Timestamp(year=row.year, month=row.day, day=row.month)
                except ValueError:
                    return row  # Geçersiz tarih oluşursa (örn: 30 Şubat) dokunma
            return row

        # Düzeltmeyi uygula
        self.df['DATE'] = self.df['DATE'].apply(fix_date_anomaly)

        # Bozukları at ve sırala
        self.df = self.df.dropna(subset=['DATE']).sort_values('DATE')
        print("✅ Tarih düzeltme tamamlandı.")

        # ---------------------------------------------------------

        # 2. EK BİLGİLERİ YEDEKLEME (METADATA)
        meta_cols = ['DATE']
        possible_cols = ['WEEKDAY', 'IS_EXAM', 'IS_SCHOOL_OPEN', 'HOLIDAY_TYPE', 'SPECIAL_EVENT']

        for col in possible_cols:
            if col in self.df.columns:
                meta_cols.append(col)

        # Her tarih için bu bilgiler sabittir, duplicate'leri atıp saklıyoruz
        self.date_context = self.df[meta_cols].drop_duplicates(subset=['DATE']).dropna(subset=['DATE'])

        # 3. Kart Tipleri Listesi
        self.card_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD',
                          'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']

    def get_institution_list(self):
        return sorted(self.df['INSTITUTION'].dropna().unique().tolist())

    def _prepare_data(self, institution, card_type):
        temp_df = self.df.copy()

        # --- KURUM FİLTRESİ ---
        if institution.lower() != "hepsi":
            temp_df = temp_df[temp_df['INSTITUTION'].str.strip().str.lower() == institution.strip().lower()]
            if temp_df.empty:
                return pd.DataFrame()

        # --- KART TİPİ SEÇİMİ ---
        target_col = 'TARGET_VALUE'

        if card_type.lower() == "hepsi":
            valid_cols = [c for c in self.card_cols if c in temp_df.columns]
            if not valid_cols: return pd.DataFrame()
            temp_df[target_col] = temp_df[valid_cols].sum(axis=1)
        else:
            col_match = None
            for col in self.card_cols:
                if col.lower() == card_type.lower():
                    col_match = col
                    break

            if col_match and col_match in temp_df.columns:
                temp_df[target_col] = temp_df[col_match]
            else:
                return pd.DataFrame()

        # --- GÜNLÜK TOPLAM ---
        daily_df = temp_df.groupby('DATE')[target_col].sum().reset_index()

        # --- METADATA BİRLEŞTİRME ---
        if not daily_df.empty:
            daily_df = pd.merge(daily_df, self.date_context, on='DATE', how='left')

        return daily_df

    def analyze(self, institution, card_type):
        """
        Grafiksiz Analiz: Sadece hesaplar, referans aralığını yazar ve hafızaya atar.
        """
        data = self._prepare_data(institution, card_type)

        if data.empty or len(data) < 10:
            return

        values = data['TARGET_VALUE']

        # --- IQR HESAPLAMA ---
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        sensitivity = 1.5

        lower_bound = max(0, Q1 - sensitivity * IQR)
        upper_bound = Q3 + sensitivity * IQR

        # Outlierları filtrele
        outliers = data[(values < lower_bound) | (values > upper_bound)].copy()

        if not outliers.empty:
            # --- 1. KONSOLA YAZDIR (GÜNCELLENDİ: REFERANS ARALIĞI EKLENDİ) ---
            print(f"\n>>> BULUNDU: {institution} - {card_type} (Adet: {len(outliers)})")
            print(f"    📏 REFERANS (NORMAL) ARALIK: {int(lower_bound):,} - {int(upper_bound):,}")

            # --- 2. HAFIZAYA KAYDET (Global Listeye Ekle) ---
            found_dates = outliers['DATE'].tolist()
            self.all_outlier_dates.extend(found_dates)

            # Detaylı satırları konsola dök
            outliers_sorted = outliers.sort_values('TARGET_VALUE', ascending=False)
            for _, row in outliers_sorted.iterrows():
                date_str = row['DATE'].strftime('%Y-%m-%d')
                val = int(row['TARGET_VALUE'])

                # Durum Belirleme (Normalin Altında mı Üstünde mi?)
                status = "YUKSEK" if val > upper_bound else "DUSUK "

                note = row['HOLIDAY_TYPE'] if 'HOLIDAY_TYPE' in row and pd.notna(row['HOLIDAY_TYPE']) else ""

                # Sınav günü kontrolü (Eğer sütun varsa)
                if 'IS_EXAM' in row and row['IS_EXAM'] == 1:
                    note = f"SINAV GUNU {note}"

                print(f"    - {date_str} | Yolcu: {val:<9,} | {status} | {note}")

    def print_final_ranking(self):
        """
        Tüm analizler bittikten sonra hafızadaki tarihleri sayar ve sıralar.
        """
        if not self.all_outlier_dates:
            print("\n[SONUC]: Hicbir analizde outlier bulunamadi.")
            return

        print("\n" + "#" * 60)
        print("🏆 GLOBAL OUTLIER SIRALAMASI (Düzeltilmiş Tarihler)")
        print("#" * 60)
        print("Bu liste, tarih düzeltmesi yapildiktan sonra en cok sorun cikaran gunleri gosterir.\n")

        # 1. Tarihleri Say
        date_counts = Counter(self.all_outlier_dates)

        # 2. DataFrame'e cevir
        ranking_df = pd.DataFrame.from_dict(date_counts, orient='index', columns=['OUTLIER_SCORE']).reset_index()
        ranking_df.rename(columns={'index': 'DATE'}, inplace=True)

        # 3. Metadata ile birlestir
        ranking_df = pd.merge(ranking_df, self.date_context, on='DATE', how='left')

        # 4. Skora gore sirala
        ranking_df = ranking_df.sort_values('OUTLIER_SCORE', ascending=False)

        # 5. Yazdir
        print(f"{'TARIH':<12} | {'SKOR':<6} | {'GUN':<10} | {'OZEL DURUM / TATIL'}")
        print("-" * 80)

        for _, row in ranking_df.iterrows():
            d_str = row['DATE'].strftime('%Y-%m-%d')
            score = row['OUTLIER_SCORE']
            day_name = row['WEEKDAY'] if 'WEEKDAY' in row and pd.notna(row['WEEKDAY']) else "-"

            reason = ""
            if 'HOLIDAY_TYPE' in row and pd.notna(row['HOLIDAY_TYPE']):
                reason = str(row['HOLIDAY_TYPE'])
            elif 'IS_EXAM' in row and row['IS_EXAM'] == 1:
                reason = "SINAV GUNU"

            print(f"{d_str:<12} | {score:<6} | {day_name:<10} | {reason}")

        print("-" * 80)
        print(f"TOPLAM UNIK OUTLIER TARIH: {len(ranking_df)}")

        # Hafızayı temizle
        self.all_outlier_dates = []

    # -------------------------------------------------------------------------


# ETKILESIMLI MENU
# -------------------------------------------------------------------------

def run_interactive_menu():
    print("\nVeri yukleniyor, lutfen bekleyin...")

    csv_path = "regression/izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"
    try:
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
    except FileNotFoundError:
        print(f"\n[HATA]: '{csv_path}' dosyasi bulunamadi.")
        return

    detector = OutlierDetector(df)
    institutions = detector.get_institution_list()
    card_types = detector.card_cols

    while True:
        print("\n" + "=" * 60)
        print(" OTOMATIK OUTLIER TARAYICI (TARIH DUZELTMELI)")
        print("=" * 60)
        print("1. [FULL TARAMA] Tum Kurumlar x Tum Kartlar")
        print("2. [GENEL]       Sadece 'Tum Izmir' Toplami")
        print("3. [KURUM]       Sadece Kurum Toplamlarini Tara")
        print("4. [KART]        Sadece Kart Tipi Toplamlarini Tara")
        print("-" * 60)
        print("Q. CIKIS")

        choice = input("\nSeciminiz (varsayilan: 1): ").strip().lower()

        if choice == 'q':
            print("Cikis yapiliyor.")
            break
        if choice == '': choice = '1'

        print("\n>>> Analiz basliyor...\n")

        if choice == '1':
            detector.analyze("Hepsi", "Hepsi")
            for inst in institutions:
                for card in card_types:
                    detector.analyze(inst, card)

        elif choice == '2':
            detector.analyze("Hepsi", "Hepsi")

        elif choice == '3':
            for inst in institutions:
                detector.analyze(inst, "Hepsi")

        elif choice == '4':
            for card in card_types:
                detector.analyze("Hepsi", card)

        else:
            print("Gecersiz secim.")
            continue

        detector.print_final_ranking()
        input("\nAna menuye donmek icin ENTER'a basin...")


if __name__ == "__main__":
    run_interactive_menu()