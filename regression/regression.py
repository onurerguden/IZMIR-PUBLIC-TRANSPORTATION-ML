import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'current-data', 'izmirim-kart-ulasim-istatistikleri-guncel.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots', 'regression')

print(f"Veri Aranıyor: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print("HATA: Dosya bulunamadı!")
    exit()

df = pd.read_csv(DATA_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. FEATURE ENGINEERING ---
df['DATE'] = pd.to_datetime(df['DATE'], format='%d.%m.%Y')
df['MONTH'] = df['DATE'].dt.month
df['WEEKDAY'] = df['DATE'].dt.weekday
df['IS_WEEKEND'] = df['WEEKDAY'].apply(lambda x: 1 if x >= 5 else 0)
df['SEASON'] = df['MONTH'].apply(lambda x: 'Winter' if x in [12, 1, 2] else ('Spring' if x in [3, 4, 5] else ('Summer' if x in [6, 7, 8] else 'Autumn')))

# Hedef Değişken
passenger_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD', 'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']
df['TOTAL_PASSENGERS'] = df[passenger_cols].sum(axis=1)

# --- KRİTİK DÜZELTME: SHIFT ---
# Bugünün tahminini yaparken bugünün verisini kullanamayız! Dünün verisini kullanmalıyız.
# shift(1) diyerek veriyi bir gün aşağı kaydırıyoruz.
# (Her kurum kendi içinde kaymalı, o yüzden groupby kullanıyoruz)
df['ROLLING_7'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(7).mean())
df['ROLLING_14'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(14).mean())
df['ROLLING_30'] = df.groupby('INSTITUTION')['TOTAL_PASSENGERS'].transform(lambda x: x.shift(1).rolling(30).mean())

# NaN temizliği (Baştaki kayıp veriler)
df = df.dropna(subset=['ROLLING_7', 'ROLLING_14', 'ROLLING_30'])

# --- 2. MODEL HAZIRLIĞI ---
# Label Encoding (Random Forest için iyidir)
le = LabelEncoder()
df['INSTITUTION_ENC'] = le.fit_transform(df['INSTITUTION'])

# X ve y Hazırla
# passenger_cols zaten target'ın parçası, onları modele verme (Leakage önleme)
drop_cols = ['DATE', 'TOTAL_PASSENGERS', '_id', 'INSTITUTION'] + passenger_cols
X = df.drop(columns=drop_cols, errors='ignore')

# Mevsim için One-Hot (Linear Regression bunu sever)
X = pd.get_dummies(X, columns=['SEASON'], drop_first=True)
y = df['TOTAL_PASSENGERS']

# Time-Series Split (%75 - %25)
train_size = int(len(df) * 0.75)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# --- 3. MODEL EĞİTİMİ ---
print("Modeller eğitiliyor...")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- 4. DEĞERLENDİRME (Filtreli) ---
# Sadece anlamlı büyüklükteki verileri (Target > 50) değerlendiriyoruz
mask = y_test > 50
y_test_valid = y_test[mask]
y_pred_lr_valid = y_pred_lr[mask]
y_pred_rf_valid = y_pred_rf[mask]

rmse_lr = np.sqrt(mean_squared_error(y_test_valid, y_pred_lr_valid))
mape_lr = mean_absolute_percentage_error(y_test_valid, y_pred_lr_valid)

rmse_rf = np.sqrt(mean_squared_error(y_test_valid, y_pred_rf_valid))
mape_rf = mean_absolute_percentage_error(y_test_valid, y_pred_rf_valid)

print("\n" + "="*40)
print("SONUÇLAR (Shift Düzeltmesi Yapıldı)")
print("="*40)
print(f"1. LINEAR REGRESSION:")
print(f"   - RMSE: {rmse_lr:,.0f}")
print(f"   - MAPE: %{mape_lr*100:.2f}")
print("-" * 40)
print(f"2. RANDOM FOREST:")
print(f"   - RMSE: {rmse_rf:,.0f}")
print(f"   - MAPE: %{mape_rf*100:.2f}")
print("="*40)

# --- 5. GRAFİK ---
plt.figure(figsize=(12, 6))
# Rastgele 50 örnek (Filtrelenmiş veriden)
sample_indices = y_test_valid.sample(50, random_state=42).index

plt.plot(range(50), y_test_valid.loc[sample_indices].values, 'o-', color='black', label='Gerçek', alpha=0.6)
plt.plot(range(50), y_pred_rf_valid[np.isin(y_test_valid.index, sample_indices)], '^--', color='green', label='Random Forest', alpha=0.8)
plt.plot(range(50), y_pred_lr_valid[np.isin(y_test_valid.index, sample_indices)], 'x--', color='red', label='Linear Reg', alpha=0.5)

plt.title('Gelişmiş Model Sonuçları (Rolling Features + Time Split)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'regression_advanced.png'))
print("Grafik kaydedildi.")


# --- ADDITIONAL PROFESSIONAL PLOTS ---

# 1. Random Forest Feature Importance Plot
importances = rf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))

# 2. Residual Plot (Random Forest)
residuals_rf = y_test_valid - y_pred_rf_valid
plt.figure(figsize=(10, 5))
plt.scatter(y_test_valid, residuals_rf, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Actual (Random Forest)")
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot_rf.png"))

# 3. Residual Histogram
plt.figure(figsize=(10, 5))
plt.hist(residuals_rf, bins=30, edgecolor='black')
plt.title("Residual Distribution (Random Forest)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_hist_rf.png"))

# 4. True vs Predicted Scatter Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test_valid, y_pred_rf_valid, alpha=0.6)
plt.plot([y_test_valid.min(), y_test_valid.max()],
         [y_test_valid.min(), y_test_valid.max()],
         'r--', linewidth=2)
plt.title("True vs Predicted (Random Forest)")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_pred_rf.png"))

print("Additional professional plots saved.")