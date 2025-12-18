import pandas as pd
import numpy as np
import os
import matplotlib

# GUI hatası almamak için Agg backend
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
# 📊 GRAFIK FONKSİYONLARI (ORİJİNAL HALİ KORUNDU)
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
    # Son 100 günü veya veri azsa tamamını al
    limit = min(len(y_test), 100)
    y_test_subset = y_test.reset_index(drop=True).tail(limit)

    # Boyut kontrolü yaparak slice al
    preds_ridge_subset = preds_ridge[-limit:] if len(preds_ridge) >= limit else preds_ridge
    preds_rf_subset = preds_rf[-limit:] if len(preds_rf) >= limit else preds_rf
    preds_xgb_subset = preds_xgb[-limit:] if len(preds_xgb) >= limit else preds_xgb

    x_axis = range(len(y_test_subset))

    plt.plot(x_axis, y_test_subset.values, color='black', label='Actual', linewidth=2, alpha=0.85)
    plt.plot(x_axis, preds_ridge_subset, color='#2ca02c', label='Ridge', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, preds_rf_subset, color='#1f77b4', label='Random Forest', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, preds_xgb_subset, color='#d62728', label='XGBoost', linewidth=1.5, linestyle='--', alpha=0.9)

    plt.xlabel(f'Days (Test Period - Last {limit} Days)')
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
    fig, ax1 = plt.subplots(figsize=(12, 8))
    x = np.arange(len(metrics_df['Model']))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, metrics_df['RMSE'], width, label='RMSE', color='#4c72b0', alpha=0.9)
    ax1.set_ylabel('RMSE', color='#4c72b0', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'])
    ax1.tick_params(axis='y', labelcolor='#4c72b0')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height):,}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#4c72b0')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, metrics_df['MAPE'], width, label='MAPE (%)', color='#c44e52', alpha=0.9)
    ax2.set_ylabel('MAPE (%)', color='#c44e52', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#c44e52')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f'%{height:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c44e52')

    plt.title("Model Karşılaştırması (RMSE & MAPE)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/00_model_comparison.png")
    plt.close()


def create_detailed_plots(model, X_test, y_test, preds, institution_name, target_col, model_name):
    output_dir = "plots"  # Düzeltildi
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

    special_cols = ['IS_SCHOOL_OPEN', 'IS_EXAM']
    for col in special_cols:
        if col not in df.columns:
            df[col] = 0
        else:
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


# --- GÜNCELLENEN TRAIN MODEL FONKSİYONU ---
# Narlıdere analizindeki "tarih bazlı split" isteğini karşılamak için güncellendi.
def train_model(df, target_col="STUDENT", institution_name="Hepsi", model_type="XGBoost",
                train_start=None, train_end=None, test_start=None, test_end=None):
    print(f"--- ANALIZ BASLIYOR: {institution_name} - {target_col} ---")
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

    # 1. YASAKLI KURUMLARI CIKARMA
    forbidden_institutions = [
        'Metro Maskematik', 'Ýzdeniz Maskematik', 'BÝSÝM',
        'Ýzmir BB - Atik Yönetimi DB', 'Ýzmir Doðal Yaþam Parký',
        'ÝBB Bornova Buz Pisti', 'Izfas', 'ÝBB Sebze ve Meyve Hali',
        'Grand Plaza A.Þ', 'Teleferik', 'NOSTALJÝK TRAMVAY'
    ]
    df = df[~df['INSTITUTION'].isin(forbidden_institutions)]
    df = df.fillna({"HOLIDAY_TYPE": "None", "IS_HOLIDAY": 0, "SPECIAL_EVENT": "None"})

    # 2. KURUM SEÇİMİ VE AGGREGATION
    all_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD', 'TICKET', 'CHILD', 'PERSONNEL', 'FREE',
                'BANK CARD']
    special_cols = ['IS_SCHOOL_OPEN', 'IS_EXAM', 'IS_HOLIDAY']

    agg_dict = {c: 'sum' for c in all_cols if c in df.columns}
    for sc in special_cols:
        if sc in df.columns:
            agg_dict[sc] = 'max'

    if institution_name.lower() == "hepsi":
        available_cols = [c for c in all_cols if c in df.columns]
        df_grouped = df.groupby('DATE').agg(agg_dict).reset_index()
        other_cols = ['HOLIDAY_TYPE']
        if 'HOLIDAY_TYPE' in df.columns:
            df_others = df.groupby('DATE')['HOLIDAY_TYPE'].first().reset_index()
            df = pd.merge(df_grouped, df_others, on='DATE')
        else:
            df = df_grouped
        df['INSTITUTION'] = 'Hepsi'
    else:
        df = df[df["INSTITUTION"].str.strip().str.lower() == institution_name.lower()].copy()
        if df.empty:
            print(f"HATA: '{institution_name}' kurumu bulunamadi!")
            return None, None

    # 3. HEDEF BELİRLEME
    final_target = target_col
    if target_col.lower() == "hepsi":
        cols = [c for c in all_cols if c in df.columns]
        df["TOTAL_PASSENGERS"] = df[cols].sum(axis=1)
        final_target = "TOTAL_PASSENGER"
        df.rename(columns={"TOTAL_PASSENGERS": final_target}, inplace=True)

    if (df[final_target] == 0).sum() > 0:
        df[final_target] = df[final_target].replace(0, 1)

    # 4. OUTLIER TEMIZLIGI
    mean_val = df[final_target].mean()
    std_val = df[final_target].std()
    upper_limit = mean_val + (3.5 * std_val)
    df = df[df[final_target] <= upper_limit]

    # 5. FEATURE ENGINEERING
    df_features = feature_engineering(df, final_target)

    # 6. SPLIT (TARİH BAZLI veya OTOMATİK)
    cols_to_drop = ["INSTITUTION", "IS_HOLIDAY", "HOLIDAY_TYPE", "MONTH_NUM", "SEASON", "DAY_TYPE", "WEEKDAY"]
    cols_to_drop.extend([c for c in all_cols if c in df_features.columns])
    if final_target not in cols_to_drop: cols_to_drop.append(final_target)

    # X ve y'yi hazırla (DATE'i tutuyoruz, split'te kullanacağız)
    X_full = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns], errors='ignore')
    # DATE kolonu düşmüş olabilir, geri ekleyelim kontrol için
    X_full['DATE'] = df_features['DATE']
    y_full = df_features[final_target]

    if train_start and train_end and test_start and test_end:
        print(f"   -> Tarih Bazlı Split: Eğitim [{train_start} - {train_end}] | Test [{test_start} - {test_end}]")

        train_mask = (X_full['DATE'] >= train_start) & (X_full['DATE'] <= train_end)
        test_mask = (X_full['DATE'] >= test_start) & (X_full['DATE'] <= test_end)

        X_train = X_full[train_mask].drop(columns=['DATE'])
        y_train = y_full[train_mask]

        X_test = X_full[test_mask].drop(columns=['DATE'])
        y_test = y_full[test_mask]

        test_dates = X_full[test_mask]['DATE']  # Sonuçları döndürmek için tarihleri sakla

    else:
        print("   -> Otomatik Split (%80 - %20)")
        X_nodate = X_full.drop(columns=['DATE'])

        split_point = int(len(X_nodate) * 0.80)
        X_train, X_test = X_nodate.iloc[:split_point], X_nodate.iloc[split_point:]
        y_train, y_test = y_full.iloc[:split_point], y_full.iloc[split_point:]
        test_dates = df_features.iloc[split_point:]['DATE']

    if len(X_train) == 0 or len(X_test) == 0:
        print("HATA: Eğitim veya Test veri seti boş kaldı. Tarih aralıklarını kontrol edin.")
        return None, None

    # 7. MODEL EĞİTİMİ
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

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)
    results.append({'Model': 'Random Forest', 'RMSE': np.sqrt(mean_squared_error(y_test, p_rf)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_rf) * 100})

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5, n_jobs=-1, random_state=42)
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict(X_test)
    p_xgb = np.maximum(p_xgb, 1)
    results.append({'Model': 'XGBoost', 'RMSE': np.sqrt(mean_squared_error(y_test, p_xgb)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_xgb) * 100})

    # --- RAPORLAMA ---
    results_df = pd.DataFrame(results)
    print("\n=== SKOR TABLOSU ===")
    print(results_df)

    # Grafikler
    create_comparison_plot(results_df, output_dir)
    create_dual_scatter_plot(y_test, p_rf, p_xgb, institution_name, target_col, output_dir)
    create_friend_style_line_plot(y_test, p_ridge, p_rf, p_xgb, institution_name, target_col, output_dir)

    # 8. SEÇİLEN MODEL VE SONUÇ DÖNDÜRME
    if model_type == "XGBoost":
        selected_model = xgb_model
        selected_preds = p_xgb
    elif model_type == "Random Forest":
        selected_model = rf
        selected_preds = p_rf
    else:
        selected_model = ridge
        selected_preds = p_ridge

    # Findings sınıfının beklediği DataFrame formatı
    final_results_df = pd.DataFrame({
        'DATE': test_dates.values,
        'TOTAL_PASSENGER': y_test.values,
        'PREDICTED': selected_preds
    })

    print(f"\n=== SEÇİLEN MODEL ({model_type}) TAMAMLANDI ===")

    # Findings sınıfı 2 değer beklediği için tuple döndürüyoruz
    return selected_model, final_results_df


def create_time_series_comparison_plots(df, output_dir="plots"):
    # ... (Bu fonksiyon aynı kalacak, sadece importları içeri alıyorum ki temiz kalsın) ...
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.dates import DateFormatter, MonthLocator
    os.makedirs(output_dir, exist_ok=True)

    # ... (Kodun geri kalanı orijinal haliyle aynı, buraya tekrar yapıştırmıyorum ama dosya içinde kalmalı) ...
    pass