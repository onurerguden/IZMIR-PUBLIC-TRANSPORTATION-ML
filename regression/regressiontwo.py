import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def clean_and_filter_data(df):
    """
    Veriyi temizler, ESHOT için filtreler ve eğitim için hazırlar.
    """
    print("--- Veri Ön İşleme ve Filtreleme Başlıyor (Sadece ESHOT) ---")

    # 1. Tarih Dönüşümü
    df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

    # 2. Temizlik
    if "HOLIDAY_TYPE" in df.columns:
        df["HOLIDAY_TYPE"] = df["HOLIDAY_TYPE"].fillna("None")
        df["HOLIDAY_TYPE"] = df["HOLIDAY_TYPE"].astype(str).replace({"nan": "None", "NaN": "None"})

    string_cols = ["DAY_TYPE", "HOLIDAY_TYPE", "INSTITUTION"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 3. Tarih Filtresi (2023 Eylül ve sonrası)
    start_date = pd.to_datetime("2023-09-01")
    df = df[df["DATE"] >= start_date].copy()

    # 4. Filtreleme
    df = df[
        (df["DAY_TYPE"].str.capitalize() == "Weekday") &
        (df["HOLIDAY_TYPE"].str.capitalize() == "None") &
        (df["INSTITUTION"] == "Eshot")
        ].copy()

    return df


def train_monthly_xgboost_student(df, target_col="STUDENT"):
    """
    Sadece Eshot verisi üzerinde aylık XGBoost modelleri eğitir.
    Ekstra olarak YÜZDESEL DOĞRULUK (ACCURACY) hesaplar.
    """
    # 1. Filtreleme
    df_clean = clean_and_filter_data(df)

    if df_clean.empty:
        raise ValueError("HATA: Filtreleme sonucunda ESHOT verisi kalmadı!")

    # 2. Sütun Seçimi
    leakage_cols = ["FULL_FARE", "TEACHER", "SIXTY_YEARS_OLD", "TICKET",
                    "CHILD", "PERSONNEL", "FREE", "BANK CARD", target_col]

    drop_cols = ["DATE", "DAY_TYPE", "HOLIDAY_TYPE", "IS_HOLIDAY", "SPECIAL_EVENT", "INSTITUTION"] + leakage_cols

    cat_cols = ["SEASON"]
    valid_cat_cols = [c for c in cat_cols if c in df_clean.columns]

    df_encoded = pd.get_dummies(df_clean, columns=valid_cat_cols, drop_first=True)

    # 3. Aylık Döngü
    models = {}
    results = []

    df_clean["MONTH_NUM"] = df_clean["DATE"].dt.month
    available_months = df_clean["MONTH_NUM"].unique()
    available_months.sort()

    print(f"\n--- {len(available_months)} Farklı Ay İçin ESHOT Model Eğitimi (Doğruluk Analizli) Başlıyor ---")

    for month in available_months:
        month_mask = df_clean["MONTH_NUM"] == month
        month_data = df_encoded[month_mask]

        X = month_data.drop(columns=[c for c in drop_cols if c in month_data.columns] + ["MONTH_NUM"], errors='ignore')
        y = df_clean.loc[month_mask, target_col]

        if len(X) < 5:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            early_stopping_rounds=10, n_jobs=-1, random_state=42
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)

        # --- METRİK HESAPLAMA ---
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        # --- [YENİ] GÜVENİLİRLİK / DOĞRULUK ALGORİTMASI ---
        # Test setindeki gerçek verilerin ortalamasını alıyoruz
        real_mean = y_test.mean()

        # Doğruluk Formülü: 1 - (Hata / Gerçek Ortalama)
        if real_mean > 0:
            error_ratio = mae / real_mean
            accuracy_percentage = (1 - error_ratio) * 100
        else:
            accuracy_percentage = 0

        # Eksiye düşerse (Hata > Gerçek) 0'a sabitle (Çok nadir olur)
        accuracy_percentage = max(0, accuracy_percentage)

        models[month] = model
        results.append({
            "Ay": int(month),
            "RMSE": round(rmse, 2),
            "MAE (Hata)": round(mae, 2),
            "Ortalama Biniş": round(real_mean, 2),
            "Doğruluk (%)": round(accuracy_percentage, 2),
            "Veri Sayısı": len(X)
        })

        print(f"Ay {month}: Eshot Modeli -> Doğruluk: %{accuracy_percentage:.2f} (Hata: {mae:.0f})")

    return models, pd.DataFrame(results)