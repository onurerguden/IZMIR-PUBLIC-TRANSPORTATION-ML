import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def clean_and_filter_data(df, institution_name="Eshot", start_date=None):
    """
    Veriyi temizler ve ISTENEN KURUM icin filtreler.
    start_date: Eger tarih verilirse (orn: '2023-07-01') o tarihten sonrasini alir.
                Eger None verilirse tarih filtresi uygulamaz (Tum veri).
    """
    print(f"--- Veri On Isleme Basliyor (Kurum: {institution_name.upper()}) ---")

    # 1. Tarih Donusumu
    df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

    # 2. Temizlik
    if "HOLIDAY_TYPE" in df.columns:
        df["HOLIDAY_TYPE"] = df["HOLIDAY_TYPE"].fillna("None")
        df["HOLIDAY_TYPE"] = df["HOLIDAY_TYPE"].astype(str).replace({"nan": "None", "NaN": "None"})

    string_cols = ["DAY_TYPE", "HOLIDAY_TYPE", "INSTITUTION"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()

    # 3. TARIH FILTRESI (Parametrik)
    if start_date is not None:
        filter_date = pd.to_datetime(start_date)
        df = df[df["DATE"] >= filter_date].copy()
        print(f"-> Tarih Filtresi Uygulandi: {start_date} ve sonrasi.")
    else:
        print("-> Tarih Filtresi KAPALI (Tum gecmis veri kullaniliyor).")

    # 4. Kurum Filtresi
    if "INSTITUTION" in df.columns:
        df = df[df["INSTITUTION"].str.lower() == institution_name.lower()].copy()

    return df


def train_general_xgboost_model(df, target_col="STUDENT", institution_name="Eshot", start_date=None):
    """
    start_date parametresini temizlik fonksiyonuna iletir.
    Leakage (Sizinti) onlemi alinmistir.
    """
    # 1. Temizlik ve Filtreleme
    df_clean = clean_and_filter_data(df, institution_name, start_date)

    if df_clean.empty:
        print(f"HATA: '{institution_name}' icin veri yok veya tarih araliginda veri bulunamadi!")
        raise ValueError("Veri seti bos.")

    # ZAMAN SIRALAMASI
    df_clean = df_clean.sort_values("DATE").reset_index(drop=True)

    print(f"{institution_name.upper()} - {target_col} icin {len(df_clean)} gun veri hazirlandi.")

    # 2. Ozellik Muhendisligi
    df_clean["MONTH_NUM"] = df_clean["DATE"].dt.month
    df_clean["DAY_OF_WEEK"] = df_clean["DATE"].dt.weekday
    df_clean["YEAR"] = df_clean["DATE"].dt.year

    # 3. Sutun Secimi (Leakage Onleme - KRITIK)
    all_card_types = ["FULL_FARE", "STUDENT", "TEACHER", "SIXTY_YEARS_OLD",
                      "TICKET", "CHILD", "PERSONNEL", "FREE", "BANK CARD"]

    # Target dahil TUM bilet sayilarini X'ten atiyoruz
    base_drop_cols = ["DATE", "INSTITUTION", "IS_HOLIDAY", "SPECIAL_EVENT"] + all_card_types

    # 4. Kategorik Kodlama
    cat_cols = ["DAY_TYPE", "HOLIDAY_TYPE", "SEASON"]
    valid_cat_cols = [c for c in cat_cols if c in df_clean.columns]

    df_encoded = pd.get_dummies(df_clean, columns=valid_cat_cols, drop_first=True)

    # 5. X ve y Ayrimi
    X = df_encoded.drop(columns=[c for c in base_drop_cols if c in df_encoded.columns], errors='ignore')
    y = df_clean[target_col]

    # 6. Train / Test Ayrimi (Shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Egitim Seti: {len(X_train)} gun | Test Seti: {len(X_test)} gun")

    # 7. Model Egitimi
    print("Model egitiliyor...")
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 8. Sonuc Hesaplama
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    real_mean = y_test.mean()

    accuracy_percentage = 0
    if real_mean > 0:
        accuracy_percentage = max(0, (1 - (mae / real_mean)) * 100)

    # Tarih araligi bilgisini rapora ekleyelim
    donem_bilgisi = f"{start_date} Sonrasi" if start_date else "Tum Tarihce"

    results = [{
        "Donem": donem_bilgisi,
        "Kurum": institution_name.upper(),
        "Kart Tipi": target_col,
        "RMSE": round(rmse, 2),
        "MAE (Hata)": round(mae, 2),
        "Ortalama Binis": round(real_mean, 2),
        "Dogruluk (%)": round(accuracy_percentage, 2),
        "Veri Sayisi": len(X_test)
    }]

    print(f"Islem Tamam! {institution_name} - {target_col} Dogruluk: %{accuracy_percentage:.2f}")

    return model, pd.DataFrame(results)