import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# --- 1. MATPLOTLIB BACKEND AYARI ---
import matplotlib

matplotlib.use('Agg')  # GUI hatası almamak için
import matplotlib.pyplot as plt
import seaborn as sns
# -----------------------------------

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# --- 2. GÖRSEL AYARLAR ---
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})


def create_residual_vs_predicted_plot(y_test, preds, model_name, output_dir, color):
    """
    İSTENİLEN GRAFİK: Residuals (Actual - Predicted) vs Predicted Values
    Revize: Başlık yok, İngilizce Eksenler.
    """
    residuals = y_test - preds

    plt.figure(figsize=(12, 7))

    # Scatter plot çizimi
    plt.scatter(preds, residuals, alpha=0.5, color=color, s=40, edgecolor='k', linewidth=0.3)

    # Sıfır noktasına referans çizgisi (Hatasız tahmin çizgisi)
    plt.axhline(0, color='black', linestyle='--', linewidth=2)

    # BAŞLIK KALDIRILDI
    # plt.title(f'Residuals vs Predicted ({model_name})', fontsize=16, fontweight='bold')

    plt.xlabel(f'Predicted Values ({model_name})', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)

    # Görseli kaydet
    filename = f"{output_dir}/Residuals_vs_Predicted_{model_name}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_friend_style_line_plot(y_test, preds_ridge, preds_rf, preds_xgb, institution_name, target_col, output_dir):
    """
    ANA GRAFİK: Son 100 Gün, 3 Model, Başlıksız, İngilizce.
    """
    plt.figure(figsize=(15, 7))

    # Sadece son 100 günü alıyoruz
    y_test_subset = y_test.reset_index(drop=True).tail(100)

    # Numpy array ise slicing, pandas ise tail kullanılır. Güvenli olması için:
    preds_ridge_subset = preds_ridge[-100:]
    preds_rf_subset = preds_rf[-100:]
    preds_xgb_subset = preds_xgb[-100:]

    # X ekseni için indeksleri resetle (0'dan 100'e kadar gitsin grafikte)
    x_axis = range(len(y_test_subset))

    plt.plot(x_axis, y_test_subset.values, color='black', label='Actual', linewidth=2, alpha=0.85)
    plt.plot(x_axis, preds_ridge_subset, color='#7f8c8d', label='Ridge', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, preds_rf_subset, color='#1f77b4', label='Random Forest', linewidth=1.5, alpha=0.8)
    plt.plot(x_axis, preds_xgb_subset, color='#d62728', label='XGBoost', linewidth=1.5, linestyle='--', alpha=0.9)

    # BAŞLIK KALDIRILDI
    # plt.title(...)

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

    # BAŞLIK KALDIRILDI
    # plt.title(...)

    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_dual_scatter.png")
    plt.close()


def create_comparison_plot(metrics_df, output_dir):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    x = np.arange(len(metrics_df['Model']))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, metrics_df['RMSE'], width, label='RMSE', color='#4c72b0', alpha=0.9)
    ax1.set_ylabel('RMSE', color='#4c72b0', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'])
    ax1.tick_params(axis='y', labelcolor='#4c72b0')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, metrics_df['MAPE'], width, label='MAPE (%)', color='#c44e52', alpha=0.9)
    ax2.set_ylabel('MAPE (%)', color='#c44e52', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#c44e52')

    def add_labels(bars, ax, is_percent=False):
        for bar in bars:
            height = bar.get_height()
            label = f'{height:.1f}%' if is_percent else f'{height:,.0f}'
            ax.text(bar.get_x() + bar.get_width() / 2., height, label,
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    add_labels(bars1, ax1)
    add_labels(bars2, ax2, is_percent=True)

    # BAŞLIK KALDIRILDI
    # plt.title("Model Performance: Error Metrics")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/00_model_comparison.png")
    plt.close()


def create_detailed_plots(model, X_test, y_test, preds, institution_name, target_col, model_name):
    output_dir = "plots"
    base_name = f"{institution_name}_{target_col}_{model_name}"

    # Residuals Histogram
    residuals = y_test - preds
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=40, kde=True, color='#1f77b4', edgecolor='black')

    # BAŞLIK KALDIRILDI
    # plt.title(...)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    plt.axvline(0, color='red', linestyle='--')
    plt.savefig(f"{output_dir}/{base_name}_residuals_hist.png")
    plt.close()

    # Timeline Zoom (Tüm zaman serisi)
    plt.figure(figsize=(15, 7))
    idx = range(len(y_test))
    plt.plot(idx, y_test.values, marker='', linestyle='-', color='black', label='Actual', linewidth=1.5, alpha=0.8)
    plt.plot(idx, preds, marker='', linestyle='--', color='#d62728', label=f'{model_name} Prediction', linewidth=1.5)

    # BAŞLIK KALDIRILDI
    # plt.title(...)
    plt.xlabel("Days (Test Period)")
    plt.ylabel("Passenger Count")

    plt.legend()
    plt.savefig(f"{output_dir}/{base_name}_timeline.png")
    plt.close()


def feature_engineering_ultimate(df, target_col):
    df = df.sort_values("DATE").copy()
    # Tarihsel featurelar
    df['MONTH_NUM'] = df['DATE'].dt.month
    df['LAG_1'] = df[target_col].shift(1)
    df['LAG_7'] = df[target_col].shift(7)
    df['ROLL_MEAN_7'] = df[target_col].shift(1).rolling(window=7).mean()
    df['ROLL_MEAN_30'] = df[target_col].shift(1).rolling(window=30).mean()

    # Trigonometrik Zaman
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['DATE'].dt.month / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['DATE'].dt.month / 12)
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DATE'].dt.weekday / 7)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DATE'].dt.weekday / 7)

    df = df.dropna(subset=['LAG_7', 'ROLL_MEAN_30'])
    return df


def train_ultimate_model(df, target_col="STUDENT", institution_name="Hepsi", model_type="XGBoost"):
    print(f"--- ANALİZ BAŞLIYOR: {institution_name} - {target_col} ---")
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. VERİ HAZIRLIĞI
    df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)
    df = df.fillna({"HOLIDAY_TYPE": "None", "IS_HOLIDAY": 0, "SPECIAL_EVENT": "None"})

    # 2. KURUM SEÇİMİ
    all_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD', 'TICKET', 'CHILD', 'PERSONNEL', 'FREE',
                'BANK CARD']
    if institution_name.lower() == "hepsi":
        available_cols = [c for c in all_cols if c in df.columns]
        df_grouped = df.groupby('DATE')[available_cols].sum().reset_index()
        df_others = df.groupby('DATE')[['IS_HOLIDAY', 'HOLIDAY_TYPE']].first().reset_index()
        df = pd.merge(df_grouped, df_others, on='DATE')
    else:
        df = df[df["INSTITUTION"].str.strip().str.lower() == institution_name.lower()].copy()

    # 3. HEDEF BELİRLEME
    final_target = target_col
    if target_col.lower() == "hepsi":
        cols = [c for c in all_cols if c in df.columns]
        df["TOTAL_PASSENGERS"] = df[cols].sum(axis=1)
        final_target = "TOTAL_PASSENGER"
        df.rename(columns={"TOTAL_PASSENGERS": final_target}, inplace=True)

    # 4. FEATURE ENGINEERING
    df_features = feature_engineering_ultimate(df, final_target)

    # 5. TEMİZLİK
    cols_to_drop = ["DATE", "INSTITUTION", "IS_HOLIDAY", "HOLIDAY_TYPE", "MONTH_NUM", "SEASON", "DAY_TYPE", "WEEKDAY"]
    card_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD', 'TICKET', 'CHILD', 'PERSONNEL', 'FREE',
                 'BANK CARD']
    cols_to_drop.extend([c for c in card_cols if c in df_features.columns])
    if final_target not in cols_to_drop: cols_to_drop.append(final_target)

    X = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns], errors='ignore')
    y = df_features[final_target]

    # 6. SPLIT
    split_point = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print("\n>>> MODELLER EĞİTİLİYOR...")
    results = []

    # --- MODEL 1: Ridge (Gri/Slate Rengi) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge = Ridge()
    ridge.fit(X_train_s, y_train)
    p_ridge = ridge.predict(X_test_s)

    results.append({'Model': 'Ridge', 'RMSE': np.sqrt(mean_squared_error(y_test, p_ridge)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_ridge) * 100})
    # İSTENEN GRAFİK: RIDGE İÇİN
    create_residual_vs_predicted_plot(y_test, p_ridge, "Ridge", output_dir, color="#2ca02c")

    # --- MODEL 2: Random Forest (Mavi Renk) ---
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)

    results.append({'Model': 'Random Forest', 'RMSE': np.sqrt(mean_squared_error(y_test, p_rf)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_rf) * 100})
    # İSTENEN GRAFİK: RF İÇİN
    create_residual_vs_predicted_plot(y_test, p_rf, "RF", output_dir, color="#1f77b4")

    # --- MODEL 3: XGBoost (Kırmızı Renk) ---
    xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=5, n_jobs=-1, random_state=42)
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict(X_test)

    results.append({'Model': 'XGBoost', 'RMSE': np.sqrt(mean_squared_error(y_test, p_xgb)),
                    'MAPE': mean_absolute_percentage_error(y_test, p_xgb) * 100})
    # İSTENEN GRAFİK: XGB İÇİN
    create_residual_vs_predicted_plot(y_test, p_xgb, "XGB", output_dir, color="#d62728")

    # --- RAPORLAMA ---
    results_df = pd.DataFrame(results)
    print("\n=== SKOR TABLOSU ===")
    print(results_df)

    create_comparison_plot(results_df, output_dir)
    create_dual_scatter_plot(y_test, p_rf, p_xgb, institution_name, target_col, output_dir)
    # Burada Ridge verisini de fonksiyona yolluyoruz
    create_friend_style_line_plot(y_test, p_ridge, p_rf, p_xgb, institution_name, target_col, output_dir)

    selected_model = xgb_model if model_type == "XGBoost" else rf
    selected_preds = p_xgb if model_type == "XGBoost" else p_rf

    print(f"\n=== SEÇİLEN MODEL ({model_type}) DETAYLARI OLUŞTURULUYOR ===")
    create_detailed_plots(selected_model, X_test, y_test, selected_preds, institution_name, target_col, model_type)

    return selected_model