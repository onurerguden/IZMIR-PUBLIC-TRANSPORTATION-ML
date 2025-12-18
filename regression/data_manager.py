import pandas as pd
import holidays
import os

# DOSYA ADIN
FILE_NAME = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"


def make_permanent_update():
    print(f"Dosya okunuyor: {FILE_NAME}...")
    df = pd.read_csv(FILE_NAME, sep=";")

    # Tarih formatini duzelt
    df['DATE'] = pd.to_datetime(df['DATE'], format='mixed', dayfirst=True)

    print("Ozel gunler hesaplaniyor...")

    # 1. OKUL TAKVIMI (Guncel)
    school_closed_periods = [
        ('2021-01-01', '2021-02-14'), ('2021-04-29', '2021-05-17'),
        ('2021-06-18', '2021-09-05'), ('2021-11-15', '2021-11-19'),
        ('2022-01-24', '2022-02-04'), ('2022-04-11', '2022-04-15'),
        ('2022-06-17', '2022-09-11'), ('2022-11-14', '2022-11-18'),
        ('2023-01-23', '2023-02-03'), ('2023-02-06', '2023-02-19'),
        ('2023-04-17', '2023-04-20'), ('2023-06-16', '2023-09-10'), ('2023-11-13', '2023-11-17'),
        ('2024-01-22', '2024-02-02'), ('2024-04-08', '2024-04-12'),
        ('2024-06-14', '2024-09-08'), ('2024-11-11', '2024-11-15'),
        ('2025-01-20', '2025-02-02'), ('2025-03-31', '2025-04-04'), ('2025-06-21', '2025-09-08')
    ]

    df['IS_SCHOOL_OPEN'] = 1
    df.loc[df['DATE'].dt.weekday >= 5, 'IS_SCHOOL_OPEN'] = 0
    for start, end in school_closed_periods:
        mask = (df['DATE'] >= pd.to_datetime(start)) & (df['DATE'] <= pd.to_datetime(end))
        df.loc[mask, 'IS_SCHOOL_OPEN'] = 0

    min_year = int(df['DATE'].dt.year.min())
    max_year = int(df['DATE'].dt.year.max())
    years = list(range(min_year, max_year + 2))
    tr_holidays = holidays.Turkey(years=years)
    df.loc[df['DATE'].apply(lambda x: x in tr_holidays), 'IS_SCHOOL_OPEN'] = 0

    # 2. SINAV GUNLERI
    exam_dates = [
        '2021-06-26', '2021-06-27', '2022-06-18', '2022-06-19',
        '2023-06-17', '2023-06-18', '2024-06-08', '2024-06-09',
        '2025-06-21', '2025-06-22', '2021-04-04', '2022-03-27',
        '2023-04-02', '2024-03-03', '2025-03-02', '2022-07-31', '2024-07-14',
        '2021-06-06', '2022-06-05', '2023-06-04', '2024-06-02', '2025-06-15'
    ]
    df['IS_EXAM'] = df['DATE'].isin(pd.to_datetime(exam_dates)).astype(int)

    # 3. AREFE GUNLERI
    unique_dates = pd.DataFrame({'DATE': df['DATE'].unique()}).sort_values('DATE')
    unique_dates['IS_HOLIDAY_TEMP'] = unique_dates['DATE'].apply(lambda x: 1 if x in tr_holidays else 0)
    unique_dates['NEXT_DAY_HOLIDAY'] = unique_dates['IS_HOLIDAY_TEMP'].shift(-1).fillna(0)
    unique_dates['IS_EVE'] = ((unique_dates['NEXT_DAY_HOLIDAY'] == 1) & (unique_dates['IS_HOLIDAY_TEMP'] == 0)).astype(
        int)

    # Eskileri sil ve yenisini ekle
    if 'IS_EVE' in df.columns: df.drop(columns=['IS_EVE'], inplace=True)
    df = pd.merge(df, unique_dates[['DATE', 'IS_EVE']], on='DATE', how='left')
    df['IS_EVE'] = df['IS_EVE'].fillna(0).astype(int)

    # KAYDETME
    print("CSV dosyasi guncelleniyor...")
    df.to_csv(FILE_NAME, sep=";", index=False)
    print(f"Basarili! '{FILE_NAME}' dosyasina yeni sutunlar eklendi.")


if __name__ == "__main__":
    make_permanent_update()