import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

# --- GORSEL AYARLAR ---
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})


# ==========================================
# 📊 GRAFIK FONKSİYONLARI (Aynı)
# ==========================================
def create_residual_vs_predicted_plot(y_test, preds, model_name, output_dir, color):
    residuals = y_test - preds
    plt.figure(figsize=(12, 7))
    plt.scatter(preds, residuals, alpha=0.5, color=color, s=40, edgecolor='k', linewidth=0.3)
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.xlabel(f'Predicted Values ({model_name})', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Residuals_vs_Predicted_{model_name}.png")
    plt.close()


def create_friend_style_line_plot(y_test, preds_ridge, preds_rf, preds_xgb, institution_name, target_col, output_dir):
    plt.figure(figsize=(15, 7))
    y_test_subset = y_test.reset_index(drop=True).tail(100)
    preds_ridge_subset = preds_ridge[-100:]
    preds_rf_subset = preds_rf[-100:]
    preds_xgb_subset = preds_xgb[-100:]
    x_axis = range(len(y_test_subset))

    plt.plot(x_axis, y_test_subset.values, color='black', label='Actual', linewidth=2, alpha=0.85)
    plt.plot(x_axis, preds_ridge_subset, color='#2ca02c', label='Ridge', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, preds_rf_subset, color='#1f77b4', label='Random Forest', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, preds_xgb_subset, color='#d62728', label='XGBoost', linewidth=1.5, linestyle='--', alpha=0.9)

    plt.xlabel('Days (Test Period - Last 100 Days)')
    plt.ylabel('Passenger Count')
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_friend_style_line_chart.png")
    plt.close()


def create_dual_scatter_plot(y_test, preds_rf, preds_xgb, institution_name, target_col, output_dir):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, preds_rf, alpha=0.4, color='#1f77b4', marker='^', s=40, label='Random Forest')
    plt.scatter(y_test, preds_xgb, alpha=0.4, color='#d62728', marker='o', s=40, label='XGBoost')
    max_val = max(y_test.max(), preds_rf.max(), preds_xgb.max())
    min_val = min(y_test.min(), preds_rf.min(), preds_xgb.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_dual_scatter.png")
    plt.close()


def create_comparison_plot(metrics_df, output_dir):
    fig, ax1 = plt.subplots(figsize=(12, 8)) # Yükseklik biraz artırıldı
    x = np.arange(len(metrics_df['Model']))
    width = 0.35

    # --- RMSE ÇUBUKLARI (Sol Eksen) ---
    bars1 = ax1.bar(x - width / 2, metrics_df['RMSE'], width, label='RMSE', color='#4c72b0', alpha=0.9)
    ax1.set_ylabel('RMSE', color='#4c72b0', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'])
    ax1.tick_params(axis='y', labelcolor='#4c72b0')

    # RMSE Değerlerini Yazdır
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{int(height):,}',  # Virgülle ayrılmış tam sayı formatı (örn: 125,000)
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#4c72b0'
        )

    # --- MAPE ÇUBUKLARI (Sağ Eksen) ---
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, metrics_df['MAPE'], width, label='MAPE (%)', color='#c44e52', alpha=0.9)
    ax2.set_ylabel('MAPE (%)', color='#c44e52', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#c44e52')

    # MAPE Değerlerini Yazdır
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'%{height:.2f}',  # Yüzdelik format (örn: %5.42)
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c44e52'
        )

    # Başlık ve Düzen
    plt.title("Model Karşılaştırması (RMSE & MAPE)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/00_model_comparison.png")
    plt.close()

def create_detailed_plots(model, X_test, y_test, preds, institution_name, target_col, model_name):
    output_dir = "plots"
    base_name = f"{institution_name}_{target_col}_{model_name}"
    residuals = y_test - preds
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=40, kde=True, color='#1f77b4', edgecolor='black')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='--')
    plt.savefig(f"{output_dir}/{base_name}_residuals_hist.png")
    plt.close()

    plt.figure(figsize=(15, 7))
    idx = range(len(y_test))
    plt.plot(idx, y_test.values, marker='', linestyle='-', color='black', label='Actual', linewidth=1.5, alpha=0.8)
    plt.plot(idx, preds, marker='', linestyle='--', color='#d62728', label=f'{model_name} Prediction', linewidth=1.5)
    plt.xlabel("Days (Test Period)")
    plt.ylabel("Passenger Count")
    plt.legend()
    plt.savefig(f"{output_dir}/{base_name}_timeline.png")
    plt.close()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if hasattr(X_test, 'columns'):
            cols = X_test.columns
        else:
            cols = [f"F{i}" for i in range(len(importances))]
        indices = np.argsort(importances)[-15:]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='#1f77b4', align='center')
        plt.yticks(range(len(indices)), [cols[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title('Top 15 Important Features')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base_name}_feature_importance.png")
        plt.close()


# ==========================================
# 🧠 VERİ İŞLEME VE MODEL
# ==========================================

def feature_engineering(df, target_col):
    df = df.sort_values("DATE").copy()

    # --- OZEL GUNLERI CSV'DEN AL ---
    # Artik hesaplamiyoruz, sadece var mi diye bakiyoruz.
    # Eger CSV guncel degilse, once update_csv.py calistirilmalidir.

    special_cols = ['IS_SCHOOL_OPEN', 'IS_EXAM', 'IS_EVE']
    for col in special_cols:
        if col not in df.columns:
            print(f"UYARI: '{col}' sutunu bulunamadi! 0 olarak varsayiliyor.")
            df[col] = 0  # Hata vermesin diye
        else:
            # NaN varsa 0 yap (Guvenlik)
            df[col] = df[col].fillna(0).astype(int)

    # 2. Lag ve Rolling
    df['MONTH_NUM'] = df['DATE'].dt.month
    df['LAG_1'] = df[target_col].shift(1)
    df['LAG_7'] = df[target_col].shift(7)
    df['ROLL_MEAN_7'] = df[target_col].shift(1).rolling(window=7).mean()
    df['ROLL_MEAN_30'] = df[target_col].shift(1).rolling(window=30).mean()

    # 3. Trigonometrik Zaman
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['DATE'].dt.month / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['DATE'].dt.month / 12)
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DATE'].dt.weekday / 7)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DATE'].dt.weekday / 7)

    df = df.dropna(subset=['LAG_7', 'ROLL_MEAN_30'])
    return df


def train_model(df, target_col="STUDENT", institution_name="Hepsi", model_type="XGBoost", start_date=None):
    print(f"--- ANALIZ BASLIYOR: {institution_name} - {target_col} ---")
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

    # Tarih Filtresi
    if start_date:
        print(f">>> Tarih Filtresi: {start_date} ve sonrasi...")
        original_len = len(df)
        df = df[df["DATE"] >= pd.to_datetime(start_date)]
        print(f"    {original_len} satirdan {len(df)} satira dusuldu.")

    # 2. YASAKLI KURUMLARI CIKARMA
    forbidden_institutions = [
        'Metro Maskematik', 'Ýzdeniz Maskematik', 'BÝSÝM',
        'Ýzmir BB - Atik Yönetimi DB', 'Ýzmir Doðal Yaþam Parký',
        'ÝBB Bornova Buz Pisti', 'Izfas', 'ÝBB Sebze ve Meyve Hali',
        'Grand Plaza A.Þ', 'Teleferik', 'NOSTALJÝK TRAMVAY'
    ]
    df = df[~df['INSTITUTION'].isin(forbidden_institutions)]

    df = df.fillna({"HOLIDAY_TYPE": "None", "IS_HOLIDAY": 0, "SPECIAL_EVENT": "None"})

    # 3. KURUM SEÇİMİ VE AGGREGATION
    all_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD', 'TICKET', 'CHILD', 'PERSONNEL', 'FREE',
                'BANK CARD']

    # Ozel gun sutunlarini korumak icin listeye ekle
    special_cols = ['IS_SCHOOL_OPEN', 'IS_EXAM', 'IS_EVE', 'IS_HOLIDAY']
    # Bu sutunlar varsa max() alarak koruyacagiz
    agg_dict = {c: 'sum' for c in all_cols if c in df.columns}
    for sc in special_cols:
        if sc in df.columns:
            agg_dict[sc] = 'max'  # Gun icinde herhangi biri 1 ise 1 olsun

    if institution_name.lower() == "hepsi":
        # A. Binis Sayilarini ve Ozel Gunleri Topla
        available_cols = [c for c in all_cols if c in df.columns]
        # agg_dict zaten hazirlandi
        df_grouped = df.groupby('DATE').agg(agg_dict).reset_index()

        # B. Diger Kategorikleri Ilk Deger Olarak Al
        other_cols = ['HOLIDAY_TYPE']
        df_others = df.groupby('DATE')[other_cols].first().reset_index()
        df = pd.merge(df_grouped, df_others, on='DATE')

        df['INSTITUTION'] = 'Hepsi'
    else:
        df = df[df["INSTITUTION"].str.strip().str.lower() == institution_name.lower()].copy()
        if df.empty: raise ValueError(f"HATA: '{institution_name}' kurumu bulunamadi!")

    # 4. HEDEF BELİRLEME
    final_target = target_col
    if target_col.lower() == "hepsi":
        cols = [c for c in all_cols if c in df.columns]
        df["TOTAL_PASSENGERS"] = df[cols].sum(axis=1)
        final_target = "TOTAL_PASSENGER"
        df.rename(columns={"TOTAL_PASSENGERS": final_target}, inplace=True)

    # --- SIFIRLARI TEMİZLEME ---
    zero_count = (df[final_target] == 0).sum()
    if zero_count > 0:
        print(f">>> UYARI: {zero_count} satirda yolcu sayisi 0. 1 yapildi.")
        df[final_target] = df[final_target].replace(0, 1)

    # 5. OUTLIER TEMIZLIGI (ASIRI YUKSEK)
    mean_val = df[final_target].mean()
    std_val = df[final_target].std()
    upper_limit = mean_val + (3.5 * std_val)

    outliers = df[df[final_target] > upper_limit]
    if len(outliers) > 0:
        print(f">>> OUTLIER: {len(outliers)} gun asiri yuksek oldugu icin cikarildi.")
        df = df[df[final_target] <= upper_limit]

    # 6. FEATURE ENGINEERING
    df_features = feature_engineering(df, final_target)

    # 7. TEMIZLIK
    cols_to_drop = ["DATE", "INSTITUTION", "IS_HOLIDAY", "HOLIDAY_TYPE", "MONTH_NUM", "SEASON", "DAY_TYPE", "WEEKDAY"]
    cols_to_drop.extend([c for c in all_cols if c in df_features.columns])
    if final_target not in cols_to_drop: cols_to_drop.append(final_target)

    X = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns], errors='ignore')
    y = df_features[final_target]

    # 8. SPLIT
    split_point = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print("\n>>> MODELLER EGITILIYOR...")
    results = []

    # Ridge
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge = Ridge()
    ridge.fit(X_train_s, y_train)
    p_ridge = ridge.predict(X_test_s)
    p_ridge = np.maximum(p_ridge, 1)
    results.append({'Model': 'Ridge', 'RMSE': np.sqrt(mean_squared_error(y_test, p_ridge)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_ridge) * 100})
    create_residual_vs_predicted_plot(y_test, p_ridge, "Ridge", output_dir, color="#2ca02c")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)
    results.append({'Model': 'Random Forest', 'RMSE': np.sqrt(mean_squared_error(y_test, p_rf)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_rf) * 100})
    create_residual_vs_predicted_plot(y_test, p_rf, "RF", output_dir, color="#1f77b4")

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5, n_jobs=-1, random_state=42)
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict(X_test)
    p_xgb = np.maximum(p_xgb, 1)
    results.append({'Model': 'XGBoost', 'RMSE': np.sqrt(mean_squared_error(y_test, p_xgb)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_xgb) * 100})
    create_residual_vs_predicted_plot(y_test, p_xgb, "XGB", output_dir, color="#d62728")

    # --- RAPORLAMA ---
    results_df = pd.DataFrame(results)
    print("\n=== SKOR TABLOSU ===")
    print(results_df)

    create_comparison_plot(results_df, output_dir)
    create_dual_scatter_plot(y_test, p_rf, p_xgb, institution_name, target_col, output_dir)
    create_friend_style_line_plot(y_test, p_ridge, p_rf, p_xgb, institution_name, target_col, output_dir)

    selected_model = xgb_model if model_type == "XGBoost" else rf
    selected_preds = p_xgb if model_type == "XGBoost" else p_rf

    r2 = r2_score(y_test, selected_preds)
    mape = mean_absolute_percentage_error(y_test, selected_preds)
    mae = mean_absolute_error(y_test, selected_preds)
    rmse = np.sqrt(mean_squared_error(y_test, selected_preds))
    real_mean = y_test.mean()

    print(f"\n=== FINAL DETAYLI RAPOR ({model_type}) ===")
    print(f"Test Seti Gunluk Ortalama: {int(real_mean)}")
    print(f"R2 Skoru: %{r2 * 100:.2f}")
    print(f"MAPE (Hata Orani): %{mape * 100:.2f}")
    print(f"Ort. Hata (MAE): {int(mae)}")
    print(f"Karesel Hata (RMSE): {int(rmse)}")

    print(f"\n=== SECILEN MODEL ({model_type}) DETAYLARI OLUSTURULUYOR ===")
    create_detailed_plots(selected_model, X_test, y_test, selected_preds, institution_name, target_col, model_type)

    return selected_model