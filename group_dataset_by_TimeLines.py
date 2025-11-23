import pandas as pd
import holidays

# --- Türkiye resmi + dini tatiller ---
turkish_holidays = holidays.Turkey()

# --- Mevsim fonksiyonu ---
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

# --- Özel eğitim/sınav dönemleri ekleme ---
def add_custom_education_events(df):
    custom_events = {
        # MEB okullarının açılış / kapanış haftaları (örnek 2025)
        "2025-09-08": "Okul Açılış Haftası",
        "2025-09-09": "Okul Açılış Haftası",
        "2025-09-10": "Okul Açılış Haftası",
        "2025-09-11": "Okul Açılış Haftası",
        "2025-09-12": "Okul Açılış Haftası",
        "2025-06-09": "Okul Kapanış Haftası",
        "2025-06-10": "Okul Kapanış Haftası",
        "2025-06-11": "Okul Kapanış Haftası",
        "2025-06-12": "Okul Kapanış Haftası",
        "2025-06-13": "Okul Kapanış Haftası",
        # ÖSYM sınavları (örnek 2025 takvimi)
        "2025-06-21": "YKS 1. Oturum TYT",
        "2025-06-22": "YKS 2. Oturum AYT",
        "2025-06-29": "KPSS Lisans",
        "2025-07-13": "DGS",
        "2025-09-07": "ALES",
    }
    df["SPECIAL_EVENT"] = df["DATE"].apply(
        lambda x: custom_events.get(x.strftime("%Y-%m-%d"), "None")
    )
    return df



df = pd.read_csv("data/current-data/izmirim-kart-ulasim-istatistikleri-guncel.csv", sep=",")
df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)


df["MONTH"] = df["DATE"].dt.month
df["SEASON"] = df["MONTH"].apply(get_season)
df["WEEKDAY"] = df["DATE"].dt.weekday
df["DAY_TYPE"] = df["WEEKDAY"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
df["IS_HOLIDAY"] = df["DATE"].apply(lambda x: 1 if x in turkish_holidays else 0)
df["HOLIDAY_TYPE"] = df["DATE"].apply(lambda x: turkish_holidays.get(x, "None"))


df = add_custom_education_events(df)


df.to_csv("izmirim-kart-ulasim-istatistikleri-guncel-extended.csv", sep=";", index=False)
print(" Yeni dosya kaydedildi: izmirim-kart-ulasim-istatistikleri-guncel-extended.csv")
