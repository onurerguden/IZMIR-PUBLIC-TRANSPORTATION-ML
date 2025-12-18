import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- AYARLAR ---
DOSYA_ADI = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 8)


class TransportVisualizer:
    def __init__(self, file_path):
        print("⏳ Veri okunuyor ve ön işleme yapılıyor...")
        try:
            self.df = pd.read_csv(file_path, sep=";", low_memory=False)
        except FileNotFoundError:
            print(f"❌ HATA: '{file_path}' dosyası bulunamadı.")
            self.df = pd.DataFrame()
            return

        # Tarih formatlama ve temizlik
        self.df['DATE'] = pd.to_datetime(self.df['DATE'], errors='coerce')
        self.df = self.df.dropna(subset=['DATE'])
        self.df = self.df[self.df['DATE'] <= '2025-05-01']

        # Kurum isimlerini standartlaştır
        self.df['INSTITUTION'] = self.df['INSTITUTION'].astype(str).str.strip().str.upper()

        # Geçerli kart tipleri listesi
        self.valid_card_types = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD',
                                 'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']

        # Mevcut kurumları temiz bir liste olarak göster
        kurumlar = sorted(self.df['INSTITUTION'].unique())
        print(f"✅ Veri yüklendi! Kurum Sayısı: {len(kurumlar)}")
        print(f"📃 Mevcut Kurumlar (İlk 5): {kurumlar[:5]} ...")

    def plot_monthly_trend(self, institution="METRO", card_type="HEPSI"):
        """
        institution: 'METRO', 'ESHOT' vb. VEYA 'HEPSI' (Tüm İzmir Toplamı)
        card_type: 'STUDENT', 'FULL_FARE' vb. VEYA 'HEPSI' (Toplam Kart), VEYA 'DETAYLI'
        """
        institution = institution.upper().strip()

        # --- 1. KURUM FİLTRESİ (GÜNCELLENDİ) ---
        if institution == "HEPSI":
            # Filtreleme yapma, tüm veriyi kullan
            subset = self.df.copy()
            title_inst = "TÜM KURUMLAR (İZMİR GENELİ)"
            print("ℹ️  Tüm kurumların verisi birleştiriliyor...")
        else:
            # Sadece seçilen kurumu al
            subset = self.df[self.df['INSTITUTION'] == institution].copy()
            title_inst = institution

            if subset.empty:
                print(f"❌ HATA: '{institution}' adında bir kurum bulunamadı.")
                print("   Lütfen listeden geçerli bir kurum adı seçin.")
                return

        # 2. AYLIK RESAMPLE (Ayın 1'ine hizala 'MS')
        numeric_cols = [c for c in self.valid_card_types if c in subset.columns]
        # groupby olmadan direkt resample yaparsak tüm satırları (tüm kurumları) toplar
        monthly_data = subset.set_index('DATE')[numeric_cols].resample('MS').sum()

        # 3. KART TİPİ SEÇİMİ VE VERİ HAZIRLIĞI
        plot_data = pd.DataFrame()
        title_suffix = ""

        if card_type == "HEPSI":
            # Tüm sütunları topla -> Tek Çizgi
            plot_data['TOPLAM YOLCU'] = monthly_data.sum(axis=1)
            title_suffix = "(Toplam Yolcu)"
            colors = ['#d62728']  # Kırmızı
            markers = ['o']

        elif card_type == "DETAYLI":
            # Hepsini ayrı ayrı çiz -> Çoklu Çizgi
            plot_data = monthly_data
            title_suffix = "(Detaylı Dağılım)"
            colors = sns.color_palette("tab10", n_colors=len(plot_data.columns))
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

        else:
            # Spesifik bir kart tipi (örn: STUDENT) -> Tek Çizgi
            if card_type in monthly_data.columns:
                plot_data[card_type] = monthly_data[card_type]
                title_suffix = f"({card_type})"
                colors = ['#1f77b4']  # Mavi
                markers = ['s']
            else:
                print(f"❌ HATA: '{card_type}' adında bir kart tipi bulunamadı.")
                return

        # 4. ÇİZİM
        if plot_data.empty or plot_data.sum().sum() == 0:
            print("⚠️ UYARI: Çizilecek veri bulunamadı veya toplam 0.")
            return

        fig, ax = plt.subplots()

        for i, col in enumerate(plot_data.columns):
            ax.plot(plot_data.index, plot_data[col],
                    marker=markers[i % len(markers)],
                    linewidth=2.5,
                    markersize=8,
                    color=colors[i],
                    label=col)

        # --- EKSEN VE GÖRÜNÜM ---
        ax.set_title(f"{title_inst} - Aylık Yolcu Trendi {title_suffix}", fontsize=16, fontweight='bold')
        ax.set_ylabel("Aylık Yolcu Sayısı", fontsize=12)
        ax.set_xlabel("Tarih", fontsize=12)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 2 ayda bir etiket
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))

        plt.grid(True, alpha=0.5, linestyle='--', which='both')
        plt.xticks(rotation=90, fontsize=10)

        # Y eksenini 'Milyon' formatında okunur yap (örn: 10,000,000)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x)))

        plt.legend(loc='upper left', title="Veri Tipi")
        plt.tight_layout()
        plt.show()


# --- KULLANIM ÖRNEKLERİ ---
if __name__ == "__main__":
    viz = TransportVisualizer(DOSYA_ADI)

    # 1. TÜM İZMİR GENELİ (Tüm kurumlar) - TOPLAM YOLCU
    # viz.plot_monthly_trend(institution="HEPSI", card_type="HEPSI")

    # 2. TÜM İZMİR GENELİ - DETAYLI (Öğrenci, Tam vs. ayrı ayrı)
    # viz.plot_monthly_trend(institution="HEPSI", card_type="DETAYLI")

    # 3. SADECE METRO - TOPLAM YOLCU
    viz.plot_monthly_trend(institution="Metro", card_type="HEPSI")