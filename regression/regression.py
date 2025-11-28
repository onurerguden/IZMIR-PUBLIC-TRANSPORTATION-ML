import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import holidays  # <-- YENİ EKLENEN KÜTÜPHANE (pip install holidays)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# --- SETTINGS & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'current-data', 'izmirim-kart-ulasim-istatistikleri-guncel.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots', 'regression')

# Grafik stili ayarları (MAVİ TEMA)
sns.set_style("whitegrid")
sns.set_palette("Blues_r")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

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

# --- YENİ EKLENEN BÖLÜM: TÜRKİYE RESMİ TATİLLERİ ---
print("Adding Turkish Public Holidays...")
# Türkiye tatillerini yükle
tr_holidays = holidays.Turkey()

# Her tarih için tatil kontrolü yap (1: Tatil, 0: Normal Gün)
df['IS_HOLIDAY'] = df['DATE'].apply(lambda x: 1 if x in tr_holidays else 0)

# İsteğe bağlı: Tatilin adını da merak edersen (Ramazan Bayramı vb.) görebilirsin
# df['HOLIDAY_NAME'] = df['DATE'].apply(lambda x: tr_holidays.get(x) if x in tr_holidays else "None")

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

# --- FEATURES LİSTESİNE IS_HOLIDAY EKLENDİ ---
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

# Ridge Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = Ridge(alpha=1.0)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=4, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- 4. EVALUATION ---
mask = y_test > 50
y_test_valid = y_test[mask]
y_pred_lr_valid = y_pred_lr[mask]
y_pred_rf_valid = y_pred_rf[mask]

rmse_lr = np.sqrt(mean_squared_error(y_test_valid, y_pred_lr_valid))
mape_lr = mean_absolute_percentage_error(y_test_valid, y_pred_lr_valid)

rmse_rf = np.sqrt(mean_squared_error(y_test_valid, y_pred_rf_valid))
mape_rf = mean_absolute_percentage_error(y_test_valid, y_pred_rf_valid)

# Overfitting Check
print("\n" + "="*60)
print("OVERFITTING CHECK (Train vs Test)")
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
print("FINAL RESULTS (Comparison)")
print("=" * 60)
print(f"1. RIDGE REGRESSION (Linear Approach):")
print(f"   - RMSE: {rmse_lr:,.0f}")
print(f"   - MAPE: %{mape_lr * 100:.2f}")
print("-" * 60)
print(f"2. RANDOM FOREST (Tree-Based Approach):")
print(f"   - RMSE: {rmse_rf:,.0f}")
print(f"   - MAPE: %{mape_rf * 100:.2f}")
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
    print(">> SONUÇ: GÜVENLİ ✅")
    print("   Modelin, sadece geçen haftayı kopyalamaktan DAHA İYİSİNİ yapıyor.")
else:
    print(">> SONUÇ: ŞÜPHELİ ⚠️")

# --- 5. VISUALIZATION ---
print("Generating professional plots...")
MAIN_BLUE = '#1f77b4'

# Plot 1: Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 8))
plt.title('Feature Importance', fontsize=16)
plt.barh(range(len(indices)), importances[indices], color=MAIN_BLUE, alpha=0.8)
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_feature_importance.png"), dpi=300)
plt.close()

# Plot 2: True vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test_valid, y_pred_rf_valid, alpha=0.5, color=MAIN_BLUE, s=50)
max_val = max(y_test_valid.max(), y_pred_rf_valid.max())
min_val = min(y_test_valid.min(), y_pred_rf_valid.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3)
plt.title('Predicted vs Actual', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_true_vs_predicted.png"), dpi=300)
plt.close()

# Plot 3: Residuals
residuals = y_test_valid - y_pred_rf_valid
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True, color=MAIN_BLUE)
plt.title('Error Distribution', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_residual_histogram.png"), dpi=300)
plt.close()

# Plot 4: Residuals vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf_valid, residuals, alpha=0.5, color=MAIN_BLUE)
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals vs Predicted', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_residuals_vs_predicted.png"), dpi=300)
plt.close()

# Plot 5: Comparison Zoomed
plt.figure(figsize=(14, 7))
sample_size = 100
plt.plot(range(sample_size), y_test_valid.values[-sample_size:], 'o-', color='black', label='Actual')
plt.plot(range(sample_size), y_pred_rf_valid[-sample_size:], '-', color=MAIN_BLUE, label='Random Forest')
plt.plot(range(sample_size), y_pred_lr_valid[-sample_size:], '--', color='#E24A33', label='Ridge Reg')
plt.title(f'Model Comparison (Last {sample_size} Days)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_model_comparison_zoomed.png"), dpi=300)
plt.close()

print(f"Plots saved to: {OUTPUT_DIR}")