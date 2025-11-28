import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import holidays
import xgboost as xgb
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# --- SETTINGS & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'current-data', 'izmirim-kart-ulasim-istatistikleri-guncel.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots', 'regression_final')

# Grafik stili ayarları (MAVİ TEMA ve MAKALE FORMATI)
sns.set_style("whitegrid")
sns.set_palette("Blues_r")
plt.rcParams.update({'font.size': 14, 'figure.figsize': (12, 8), 'font.family': 'sans-serif'})

print(f"Searching for data at: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"ERROR: File not found at {DATA_PATH}")
    exit()

# --- 1. DATA LOADING & PREPROCESSING ---
print("Loading and preprocessing data...")

try:
    with open(DATA_PATH, 'r') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','
except:
    sep = ','

df = pd.read_csv(DATA_PATH, sep=sep)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tarih Formatı Düzeltme
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')

if df['DATE'].isna().sum() > 0:
    print(f"UYARI: {df['DATE'].isna().sum()} satırda tarih hatası var ve silindi.")
    df = df.dropna(subset=['DATE'])

# Sıralama
df = df.sort_values(by=['DATE', 'INSTITUTION'])

# Date Features
df['MONTH'] = df['DATE'].dt.month
df['WEEKDAY'] = df['DATE'].dt.weekday
df['IS_WEEKEND'] = df['WEEKDAY'].apply(lambda x: 1 if x >= 5 else 0)
df['SEASON'] = df['MONTH'].apply(lambda x: 'Winter' if x in [12, 1, 2] else (
    'Spring' if x in [3, 4, 5] else ('Summer' if x in [6, 7, 8] else 'Autumn')))

# --- TÜRKİYE RESMİ TATİLLERİ ---
print("Adding Turkish Public Holidays...")
tr_holidays = holidays.Turkey()
df['IS_HOLIDAY'] = df['DATE'].apply(lambda x: 1 if x in tr_holidays else 0)
print(f">> Toplam {df['IS_HOLIDAY'].sum()} adet tatil günü işaretlendi.")

# Target Variable Creation
passenger_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD', 'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']
available_passenger_cols = [c for c in passenger_cols if c in df.columns]
df['TOTAL_PASSENGERS'] = df[available_passenger_cols].sum(axis=1)

# --- FEATURE ENGINEERING ---
print("Engineering lag features...")
df['LAG_1'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].shift(1)
df['LAG_7'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].shift(7)

df['ROLLING_7_MEAN'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(7).mean())
df['ROLLING_30_MEAN'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(30).mean())

df = df.dropna(subset=['LAG_1', 'LAG_7', 'ROLLING_7_MEAN', 'ROLLING_30_MEAN'])

# --- 2. DATA SPLITTING & PREPARATION ---
train_size = int(len(df) * 0.80)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

le_inst = LabelEncoder()
train_df['INSTITUTION_ENC'] = le_inst.fit_transform(train_df['INSTITUTION'])
test_df['INSTITUTION_ENC'] = test_df['INSTITUTION'].apply(
    lambda x: le_inst.transform([x])[0] if x in le_inst.classes_ else -1)

# --- FEATURES LİSTESİ ---
features = ['INSTITUTION_ENC', 'MONTH', 'WEEKDAY', 'IS_WEEKEND', 'IS_HOLIDAY', 'LAG_1', 'LAG_7', 'ROLLING_7_MEAN', 'ROLLING_30_MEAN']

train_df = pd.get_dummies(train_df, columns=['SEASON'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['SEASON'], drop_first=True)

for col in train_df.columns:
    if col not in test_df.columns:
        test_df[col] = 0

feature_cols = [c for c in train_df.columns if c in features or 'SEASON_' in c]

X_train = train_df[feature_cols]
y_train = train_df['TOTAL_PASSENGERS']
X_test = test_df[feature_cols]
y_test = test_df['TOTAL_PASSENGERS']

# --- 3. MODEL TRAINING ---
print("Training models...")

# MODEL 1: Ridge Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = Ridge(alpha=1.0)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# MODEL 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=4, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# MODEL 3: XGBoost
print("Training XGBoost Model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# --- 4. EVALUATION ---
mask = y_test > 50
y_test_valid = y_test[mask]

# Predictions on valid data
y_pred_lr_valid = y_pred_lr[mask]
y_pred_rf_valid = y_pred_rf[mask]
y_pred_xgb_valid = y_pred_xgb[mask]

# Metrics Calculation
rmse_lr = np.sqrt(mean_squared_error(y_test_valid, y_pred_lr_valid))
mape_lr = mean_absolute_percentage_error(y_test_valid, y_pred_lr_valid)

rmse_rf = np.sqrt(mean_squared_error(y_test_valid, y_pred_rf_valid))
mape_rf = mean_absolute_percentage_error(y_test_valid, y_pred_rf_valid)

rmse_xgb = np.sqrt(mean_squared_error(y_test_valid, y_pred_xgb_valid))
mape_xgb = mean_absolute_percentage_error(y_test_valid, y_pred_xgb_valid)

# Overfitting Check (Random Forest)
print("\n" + "="*60)
print("OVERFITTING CHECK (Random Forest: Train vs Test)")
print("="*60)
y_train_pred = rf.predict(X_train)
mask_train = y_train > 50
y_train_valid = y_train[mask_train]
y_train_pred_valid = y_train_pred[mask_train]
mape_train = mean_absolute_percentage_error(y_train_valid, y_train_pred_valid)

print(f"Random Forest TRAIN MAPE: %{mape_train*100:.2f}")
print(f"Random Forest TEST  MAPE: %{mape_rf*100:.2f}")

diff = (mape_rf - mape_train) * 100
print(f"Difference: %{diff:.2f}")

if diff < 5:
    print(">> SONUÇ: Mükemmel Denge! (No Overfitting)")
elif diff < 10:
    print(">> SONUÇ: Kabul Edilebilir Fark (Slight Overfitting - Normal)")
else:
    print(">> UYARI: Yüksek Fark! Model ezberliyor olabilir (High Overfitting)")
print("="*60)


print("\n" + "=" * 60)
print("FINAL RESULTS (Comparison of 3 Models)")
print("=" * 60)
print(f"1. RIDGE REGRESSION (Linear Approach):")
print(f"   - RMSE: {rmse_lr:,.0f}")
print(f"   - MAPE: %{mape_lr * 100:.2f}")
print("-" * 60)
print(f"2. RANDOM FOREST (Bagging Approach):")
print(f"   - RMSE: {rmse_rf:,.0f}")
print(f"   - MAPE: %{mape_rf * 100:.2f}")
print("-" * 60)
print(f"3. XGBOOST:")
print(f"   - RMSE: {rmse_xgb:,.0f}")
print(f"   - MAPE: %{mape_xgb * 100:.2f}")
print("=" * 60)


# Leakage Check
print("\n" + "="*60)
print("LEAKAGE & BASELINE KONTROLÜ")
print("="*60)
naive_forecast = X_test['LAG_7']
mask_naive = y_test > 50
rmse_naive = np.sqrt(mean_squared_error(y_test[mask_naive], naive_forecast[mask_naive]))
mape_naive = mean_absolute_percentage_error(y_test[mask_naive], naive_forecast[mask_naive])

print(f"NAIVE MODEL (Sadece Geçen Haftayı Kopyala):")
print(f"   - MAPE: %{mape_naive*100:.2f}")
print("-" * 60)
print(f"SENİN MODELİN (Random Forest):")
print(f"   - MAPE: %{mape_rf*100:.2f}")
print("-" * 60)

if mape_rf < mape_naive:
    print(">> SONUÇ: GÜVENLİ ")
    print("   Modelin, sadece geçen haftayı kopyalamaktan DAHA İYİSİNİ yapıyor.")
else:
    print(">> SONUÇ: ŞÜPHELİ ")

# --- 5. VISUALIZATION (UPDATED) ---
print("Generating professional plots...")

# Renkler
COLOR_RIDGE = '#95a5a6' # Gri
COLOR_RF = '#1f77b4'    # Mavi
COLOR_XGB = '#d62728'   # Kırmızı/Turuncu

# Residual Hesaplamaları
residuals_ridge = y_test_valid - y_pred_lr_valid
residuals_rf = y_test_valid - y_pred_rf_valid
residuals_xgb = y_test_valid - y_pred_xgb_valid

# --- PLOT 00-A: Feature Importance (Random Forest) ---
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices_rf)), importances_rf[indices_rf], color=COLOR_RF, alpha=0.8)
plt.yticks(range(len(indices_rf)), [X_train.columns[i] for i in indices_rf])
plt.xlabel("Importance Score (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "00_a_feature_importance_rf.png"), dpi=300)
plt.close()

# --- PLOT 00-B: Feature Importance (XGBoost) ---
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], color=COLOR_XGB, alpha=0.8)
plt.yticks(range(len(indices_xgb)), [X_train.columns[i] for i in indices_xgb])
plt.xlabel("Importance Score (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "00_b_feature_importance_xgb.png"), dpi=300)
plt.close()

# --- PLOT 2: True vs Predicted ---
plt.figure(figsize=(10, 8))
plt.scatter(y_test_valid, y_pred_rf_valid, alpha=0.4, color=COLOR_RF, s=40, label='Random Forest')
plt.scatter(y_test_valid, y_pred_xgb_valid, alpha=0.4, color=COLOR_XGB, s=20, label='XGBoost')
max_val = max(y_test_valid.max(), y_pred_rf_valid.max())
min_val = min(y_test_valid.min(), y_pred_rf_valid.min())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Values (Passenger Count)')
plt.ylabel('Predicted Values (Passenger Count)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_true_vs_predicted_comparison.png"), dpi=300)
plt.close()

# --- PLOT 3: Zoomed Time Series (Markers Added!) ---
plt.figure(figsize=(15, 8))
sample_size = 100
plt.plot(range(sample_size), y_test_valid.values[-sample_size:], marker='o', markersize=6, linestyle='-', color='black', label='Actual Data', linewidth=2)
# Random Forest
plt.plot(range(sample_size), y_pred_rf_valid[-sample_size:], marker='^', markersize=5, linestyle='-', color=COLOR_RF, label='Random Forest', linewidth=1.5, alpha=0.7)
# XGBoost
plt.plot(range(sample_size), y_pred_xgb_valid[-sample_size:], marker='s', markersize=4, linestyle='--', color=COLOR_XGB, label='XGBoost', linewidth=1.5, alpha=0.9)
# Ridge
plt.plot(range(sample_size), y_pred_lr_valid[-sample_size:], marker='x', markersize=4, linestyle=':', color=COLOR_RIDGE, label='Ridge', alpha=0.6)

plt.xlabel('Days (Sample Index)')
plt.ylabel('Passenger Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_model_comparison_zoomed_markers.png"), dpi=300)
plt.close()

# --- PLOT 4: Random Forest Residuals ---
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf_valid, residuals_rf, alpha=0.5, color=COLOR_RF, s=25)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values (RF)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_residuals_random_forest.png"), dpi=300)
plt.close()

# --- PLOT 5: XGBoost Residuals ---
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_xgb_valid, residuals_xgb, alpha=0.5, color=COLOR_XGB, s=25)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values (XGB)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_residuals_xgboost.png"), dpi=300)
plt.close()

# --- PLOT 6: Ridge Residuals ---
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr_valid, residuals_ridge, alpha=0.5, color=COLOR_RIDGE, s=25)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values (Ridge)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_residuals_ridge.png"), dpi=300)
plt.close()

# --- PLOT 7: Error Analysis (Histogram Distribution) ---
plt.figure(figsize=(10, 6))
sns.histplot(residuals_xgb, kde=True, color=COLOR_XGB, label='XGBoost Residuals', bins=30, alpha=0.6)
sns.histplot(residuals_rf, kde=True, color=COLOR_RF, label='Random Forest Residuals', bins=30, alpha=0.3)
plt.xlabel('Error (Residuals)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_error_analysis_distribution.png"), dpi=300)
plt.close()

# --- PLOT 8: Total Accuracy Comparison (Bar Chart) ---
metrics_df = pd.DataFrame({
    'Model': ['Ridge', 'Random Forest', 'XGBoost'],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
    'MAPE': [mape_lr*100, mape_rf*100, mape_xgb*100]
})

fig, ax1 = plt.subplots(figsize=(12, 7))

x = np.arange(len(metrics_df['Model']))
width = 0.35

# RMSE Barları
bars1 = ax1.bar(x - width/2, metrics_df['RMSE'], width, label='RMSE', color='#4c72b0', alpha=0.9)
ax1.set_ylabel('RMSE (Passenger Count)', color='#4c72b0', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#4c72b0')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_df['Model'])

# MAPE Barları
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, metrics_df['MAPE'], width, label='MAPE (%)', color='#c44e52', alpha=0.9)
ax2.set_ylabel('MAPE (%)', color='#c44e52', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#c44e52')

def add_labels(bars, ax, is_percent=False):
    for bar in bars:
        height = bar.get_height()
        label = f'{height:.1f}%' if is_percent else f'{height:.0f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

add_labels(bars1, ax1, is_percent=False)
add_labels(bars2, ax2, is_percent=True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_total_accuracy_comparison.png"), dpi=300)
plt.close()

# --- BONUS PLOT 9: Q-Q Plot (XGBoost) ---
plt.figure(figsize=(10, 8))
stats.probplot(residuals_xgb, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color(COLOR_XGB)
plt.gca().get_lines()[0].set_markerfacecolor(COLOR_XGB)
plt.gca().get_lines()[1].set_color('black')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values (Residuals)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_qq_plot_xgb.png"), dpi=300)
plt.close()

print(f"All updated plots saved to: {OUTPUT_DIR}")