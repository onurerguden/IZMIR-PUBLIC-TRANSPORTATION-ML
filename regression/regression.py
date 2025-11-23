import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Grafikleri güzelleştirmek için
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# --- SETTINGS & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'current-data', 'izmirim-kart-ulasim-istatistikleri-guncel.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots', 'regression')

# Grafik stili ayarları (MAVİ TEMA İÇİN)
sns.set_style("whitegrid")
sns.set_palette("Blues_r") # Varsayılan paleti Mavi tonları yapıyoruz
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

print(f"Searching for data at: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print("ERROR: File not found!")
    exit()

# --- 1. DATA LOADING & PREPROCESSING ---
df = pd.read_csv(DATA_PATH)
# Klasör yoksa oluştur, varsa içindekilerin üzerine yazılacak zaten
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date Conversion
df['DATE'] = pd.to_datetime(df['DATE'], format='%d.%m.%Y')
df['MONTH'] = df['DATE'].dt.month
df['WEEKDAY'] = df['DATE'].dt.weekday
df['IS_WEEKEND'] = df['WEEKDAY'].apply(lambda x: 1 if x >= 5 else 0)
df['SEASON'] = df['MONTH'].apply(lambda x: 'Winter' if x in [12, 1, 2] else ('Spring' if x in [3, 4, 5] else ('Summer' if x in [6, 7, 8] else 'Autumn')))

# Target Variable Creation
passenger_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD', 'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']
df['TOTAL_PASSENGERS'] = df[passenger_cols].sum(axis=1)

# --- FEATURE ENGINEERING (The Critical Part: Shift & Rolling) ---
df['ROLLING_7'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(7).mean())
df['ROLLING_14'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(14).mean())
df['ROLLING_30'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(30).mean())

# Drop NaN values created by rolling/shifting
df = df.dropna(subset=['ROLLING_7', 'ROLLING_14', 'ROLLING_30'])

# --- 2. MODEL PREPARATION ---
# Label Encoding for Institution (Good for Random Forest)
le = LabelEncoder()
df['INSTITUTION_ENC'] = le.fit_transform(df['INSTITUTION'])

# Prepare X and y
drop_cols = ['DATE', 'TOTAL_PASSENGERS', '_id', 'INSTITUTION'] + passenger_cols
X = df.drop(columns=drop_cols, errors='ignore')

# One-Hot Encoding for Season
X = pd.get_dummies(X, columns=['SEASON'], drop_first=True)
y = df['TOTAL_PASSENGERS']

# Time-Series Split (No Shuffling) - 75% Train, 25% Test
train_size = int(len(df) * 0.75)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# --- 3. MODEL TRAINING ---
print("Training models...")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- 4. EVALUATION ---
# Filter out very small values (<50) to stabilize MAPE
mask = y_test > 50
y_test_valid = y_test[mask]
y_pred_lr_valid = y_pred_lr[mask]
y_pred_rf_valid = y_pred_rf[mask]

# Calculate Metrics
rmse_lr = np.sqrt(mean_squared_error(y_test_valid, y_pred_lr_valid))
mape_lr = mean_absolute_percentage_error(y_test_valid, y_pred_lr_valid)

rmse_rf = np.sqrt(mean_squared_error(y_test_valid, y_pred_rf_valid))
mape_rf = mean_absolute_percentage_error(y_test_valid, y_pred_rf_valid)

print("\n" + "="*50)
print("FINAL RESULTS (With Rolling Features & Shift Correction)")
print("="*50)
print(f"1. LINEAR REGRESSION:")
print(f"   - RMSE: {rmse_lr:,.0f}")
print(f"   - MAPE: %{mape_lr*100:.2f}")
print("-" * 50)
print(f"2. RANDOM FOREST:")
print(f"   - RMSE: {rmse_rf:,.0f}")
print(f"   - MAPE: %{mape_rf*100:.2f}")
print("="*50)

# --- 5. PROFESSIONAL VISUALIZATION (MAVİ TEMA) ---

print("Generating plots...")

# Common Blue Color Code (Dodger Blue / Steel Blue Style)
MAIN_BLUE = '#1f77b4'  # Klasik Seaborn/Matplotlib Mavisi

# PLOT 1: Feature Importance (Sorted) - BLUE BARS
# ------------------------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances) # Sort indices
plt.figure(figsize=(10, 8))
plt.title('Feature Importance: What Drives Passenger Numbers?', fontsize=16)
# Bar rengini Mavi yaptık
plt.barh(range(len(indices)), importances[indices], color=MAIN_BLUE, align='center', alpha=0.8)
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance (0-1)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_feature_importance.png"), dpi=300)
plt.close()

# PLOT 2: True vs Predicted (Scatter Plot) - BLUE DOTS
# ------------------------------------------------
plt.figure(figsize=(10, 8))
# Nokta rengini Mavi yaptık
plt.scatter(y_test_valid, y_pred_rf_valid, alpha=0.6, color=MAIN_BLUE, edgecolor='k', s=60)
# Identity Line
max_val = max(y_test_valid.max(), y_pred_rf_valid.max())
min_val = min(y_test_valid.min(), y_pred_rf_valid.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Fit ($y=x$)')
plt.title('Random Forest Accuracy: Predicted vs Actual', fontsize=16)
plt.xlabel('Actual Passenger Count')
plt.ylabel('Predicted Passenger Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_true_vs_predicted.png"), dpi=300)
plt.close()

# PLOT 3: Residual Distribution (Histogram) - BLUE BARS
# ------------------------------------------------
residuals = y_test_valid - y_pred_rf_valid
plt.figure(figsize=(10, 6))
# Histogram rengini Mavi yaptık
sns.histplot(residuals, bins=40, kde=True, color=MAIN_BLUE, edgecolor='black')
plt.title('Distribution of Prediction Errors (Residuals)', fontsize=16)
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--') # Zero error line
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_residual_histogram.png"), dpi=300)
plt.close()

# PLOT 4: Residuals vs Predicted - BLUE DOTS
# ------------------------------------------------
plt.figure(figsize=(10, 6))
# Nokta rengini Mavi yaptık
plt.scatter(y_pred_rf_valid, residuals, alpha=0.6, color=MAIN_BLUE, edgecolor='k', s=50)
plt.axhline(0, color='black', linestyle='--', lw=2)
plt.title('Residuals vs Predicted Values', fontsize=16)
plt.xlabel('Predicted Passenger Count')
plt.ylabel('Residuals (Error)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_residuals_vs_predicted.png"), dpi=300)
plt.close()

# PLOT 5: Model Comparison - RF is BLUE (Hero), LR is ORANGE (Contrast)
# ------------------------------------------------
plt.figure(figsize=(14, 7))
sample_idx = y_test_valid.sample(60, random_state=42).index
sample_true = y_test_valid.loc[sample_idx].values
sample_rf = y_pred_rf_valid[np.isin(y_test_valid.index, sample_idx)]
sample_lr = y_pred_lr_valid[np.isin(y_test_valid.index, sample_idx)]

plt.plot(sample_true, 'o-', color='black', label='Actual Data', linewidth=2, markersize=8)
# Random Forest (Bizim Model) MAVİ
plt.plot(sample_rf, '^--', color=MAIN_BLUE, label='Random Forest (Best)', linewidth=2.5, markersize=8)
# Linear Regression (Kötü Olan) TURUNCU/KIRMIZI (Kontrast olsun diye)
plt.plot(sample_lr, 'x:', color='#E24A33', label='Linear Regression', linewidth=1.5, markersize=8, alpha=0.7)

plt.title('Model Comparison: Random Forest vs Linear Regression (Sample)', fontsize=16)
plt.xlabel('Sample Days (Index)')
plt.ylabel('Passenger Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_model_comparison_zoomed.png"), dpi=300)
plt.close()

print(f"All 5 Blue-Themed professional plots saved to: {OUTPUT_DIR}")