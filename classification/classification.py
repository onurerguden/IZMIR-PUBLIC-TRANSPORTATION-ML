import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import tree
import warnings
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (14, 8)})


def prepare_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, sep=";")

    # 1. Date Formatting
    try:
        df["DATE"] = pd.to_datetime(df["DATE"], format='%d.%m.%Y')
    except:
        df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

    # 2. Total Passengers Calculation
    passenger_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD',
                      'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']

    for col in passenger_cols:
        if col not in df.columns:
            df[col] = 0

    df['TOTAL_PASSENGERS'] = df[passenger_cols].sum(axis=1)

    # Sort for lag calculation
    df = df.sort_values(['INSTITUTION', 'DATE']).reset_index(drop=True)
    return df


def feature_engineering(df, target_col='TOTAL_PASSENGERS'):
    print("Feature Engineering (One-Hot Encoding & Lags)...")

    # === 1. CALENDAR FEATURES ===
    df['MONTH'] = df['DATE'].dt.month
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)

    # Cyclical encoding
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)

    # === 2. LAG FEATURES (Crucial History) ===
    # Lag 1: Yesterday, Lag 7: Last Week
    df['LAG_1'] = df.groupby('INSTITUTION')[target_col].shift(1)
    df['LAG_7'] = df.groupby('INSTITUTION')[target_col].shift(7)
    df['ROLL_MEAN_7'] = df.groupby('INSTITUTION')[target_col].shift(1).rolling(7).mean().reset_index(0, drop=True)

    # === 3. TARGET CREATION ===
    q1 = df[target_col].quantile(0.33)
    q2 = df[target_col].quantile(0.66)

    def get_class(x):
        if x <= q1:
            return 'LOW'
        elif x <= q2:
            return 'MEDIUM'
        else:
            return 'HIGH'

    df['PASSENGER_LEVEL'] = df[target_col].apply(get_class)
    df = df.dropna()
    return df


def train_and_compare_models_optimized(df):
    # === FEATURE SELECTION ===
    # We will use One-Hot Encoding for Categorical Data now
    # This creates more columns but makes distance-based models (KNN, SVM) MUCH smarter.

    categorical_cols = ['INSTITUTION', 'IS_HOLIDAY']
    numerical_cols = ['MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS',
                      'IS_WEEKEND', 'LAG_1', 'LAG_7', 'ROLL_MEAN_7']

    # Create X with One-Hot Encoding
    X = pd.get_dummies(df[categorical_cols + numerical_cols], columns=categorical_cols, drop_first=True)
    y = df['PASSENGER_LEVEL']

    # Encode Target
    from sklearn.preprocessing import LabelEncoder
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)

    # === SPLIT ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    # === SCALING (ROBUST SCALER) ===
    # RobustScaler is better than StandardScaler if there are outliers (extreme passenger days)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === OPTIMIZED MODELS ===
    models = {
        # 1. Decision Tree: Keep as is, it was already good.
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),

        # 2. SVM: Increased 'C' to 10 (less regularization, fits data tighter)
        #    Using 'rbf' kernel is standard, gamma='scale' adapts to feature variance.
        "SVM (Optimized)": SVC(kernel='rbf', C=5, gamma='scale', random_state=42, class_weight='balanced'),

        # 3. Naive Bayes: Usually fixed, but scaling helps.
        "Naive Bayes": GaussianNB(),

        # 4. KNN: Changed weights to 'distance'.
        #    This means "Closer neighbors vote more". Very effective!
        #    Reduced neighbors to 5 for sharper boundaries.
        "KNN (Optimized)": KNeighborsClassifier(n_neighbors=15, weights='distance', metric='manhattan')
    }

    results = {}

    print("\n" + "=" * 40)
    print("OPTIMIZED MODEL PERFORMANCE")
    print("=" * 40)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n {name}")
        print(f"   Accuracy: %{acc * 100:.2f}")

    # === DECISION TREE VISUALIZATION (High Quality for Paper) ===
    if "Decision Tree" in models:
        dt_model = models["Decision Tree"]
        plt.figure(figsize=(22, 14))
        tree.plot_tree(
            dt_model,
            feature_names=X.columns,
            class_names=le_target.classes_,
            filled=True,
            rounded=True,
            fontsize=8
        )

        plt.tight_layout()
        dt_output_file = "decision_tree_visualization.png"
        plt.savefig(dt_output_file, dpi=500, bbox_inches='tight')
        print(f"\n Decision Tree image saved as '{dt_output_file}' (High DPI for publication)")
        plt.close()

    # === VISUALIZATION ===
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results.keys(), results.values(), color=['#3498db', '#9b59b6', '#2ecc71', '#e67e22'])

    plt.ylim(0, 1.1)
    plt.title('Optimized Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'%{height * 100:.1f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = "algorithm_comparison_optimized.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n Graph saved as '{output_file}'")

    return results


def train_and_compare_models_time_series(df):

    print("\n=== Time-Series Split & Leakage-Free Training ===")

    # =======================
    #  Time Series Split
    # =======================
    split_date = df['DATE'].quantile(0.80)  # %80 train, %20 test (Zamana göre)
    train = df[df['DATE'] <= split_date].copy()
    test = df[df['DATE'] > split_date].copy()

    print(f"Train date range: {train['DATE'].min()} -> {train['DATE'].max()}")
    print(f"Test date range:  {test['DATE'].min()} -> {test['DATE'].max()}")

    # =======================
    #  Quantile Thresholds (Only Train)
    # =======================
    q1 = train['TOTAL_PASSENGERS'].quantile(0.33)
    q2 = train['TOTAL_PASSENGERS'].quantile(0.66)

    def get_class(x):
        if x <= q1:
            return 'LOW'
        elif x <= q2:
            return 'MEDIUM'
        else:
            return 'HIGH'

    train['PASSENGER_LEVEL'] = train['TOTAL_PASSENGERS'].apply(get_class)
    test['PASSENGER_LEVEL'] = test['TOTAL_PASSENGERS'].apply(get_class)

    # =======================
    #  Lag & Rolling Calculation - Only Train History
    # =======================
    def create_lags(data):
        data['LAG_1'] = data.groupby('INSTITUTION')['TOTAL_PASSENGERS'].shift(1)
        data['LAG_7'] = data.groupby('INSTITUTION')['TOTAL_PASSENGERS'].shift(7)
        data['ROLL_MEAN_7'] = data.groupby('INSTITUTION')['TOTAL_PASSENGERS'].shift(1).rolling(7).mean()
        return data

    train = create_lags(train)
    test = create_lags(test)

    # Feature drop (avoid NaNs)
    train = train.dropna()
    test = test.dropna()

    # =======================
    # One-Hot Encoding
    # =======================
    categorical_cols = ['INSTITUTION', 'IS_HOLIDAY']
    numerical_cols = ['MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS',
                      'IS_WEEKEND', 'LAG_1', 'LAG_7', 'ROLL_MEAN_7']

    X_train = pd.get_dummies(train[categorical_cols + numerical_cols], drop_first=True)
    X_test = pd.get_dummies(test[categorical_cols + numerical_cols], drop_first=True)

    # Align Columns (Important!)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(train['PASSENGER_LEVEL'])
    y_test = le.transform(test['PASSENGER_LEVEL'])

    # =======================
    #  Scale (Only Train -> Apply to Test)
    # =======================
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =======================
    #  Models
    # =======================
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', C=5, class_weight='balanced', random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights='distance')
    }

    results = {}

    print("\n=== PERFORMANCE RESULTS (Leakage-Free) ===")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name}: %{acc*100:.2f}")

    # =======================
    #  Decision Tree Visualization (Publication Quality)
    # =======================
    if "Decision Tree" in models:
        dt_model = models["Decision Tree"]
        plt.figure(figsize=(20, 12))

        # plot_tree parametreleri:
        # filled=True -> renk ile sınıf ayrımı
        # rounded=True -> node kenarları oval
        # fontsize -> okunabilir
        # precision -> sayısal değerleri yuvarlama
        # proportion -> node içindeki oranları gösterme
        tree.plot_tree(
            dt_model,
            feature_names=X_train.columns,
            class_names=le.classes_,
            filled=True,
            rounded=True,
            fontsize=10,
            precision=2,
            proportion=False,
            label='all'
        )

        plt.axis('off')  # Eksik gereksiz eksenleri kaldır
        plt.tight_layout()
        dt_output_file = "decision_tree_publication.png"
        plt.savefig(dt_output_file, dpi=600, bbox_inches='tight')  # yüksek çözünürlük
        print(f"Decision Tree image saved as '{dt_output_file}'")
        plt.close()
    # =======================
    #  Bar Plot Comparison
    # =======================
    plt.figure(figsize=(12, 6))
    # Çok Koyu Lacivert -> Kobalt -> Gök Mavisi -> Pastel Mavi
    bars = plt.bar(results.keys(), results.values(),
                   color=['#002060', '#0047AB', '#0070C0', '#5B9BD5'],
                   edgecolor='black', linewidth=1)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'%{height*100:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = "algorithm_comparison_timeseries.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved as '{output_file}'")


    plot_all_confusion_matrices(models, X_test_scaled, y_test, le)
    return results



def plot_all_confusion_matrices(models_dict, X_test, y_test, label_encoder, output_folder='confusion_plots'):
    """
    Sözlükteki (models_dict) TÜM modeller için normalize edilmiş
    Confusion Matrix çizer ve klasöre kaydeder.

    Parametreler:
    - models_dict: {'Model Adı': model_objesi} şeklindeki sözlük.
    - X_test: Test verisi (scaled/işlenmiş).
    - y_test: Gerçek test etiketleri.
    - label_encoder: Sınıf isimlerini (HIGH, LOW, MEDIUM) almak için.
    - output_folder: Grafiklerin kaydedileceği klasör adı.
    """

    # 1. Klasör yoksa oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' klasörü oluşturuldu.")

    classes = label_encoder.classes_

    # 2. Döngü: Her model için işlem yap
    for model_name, model in models_dict.items():
        print(f" {model_name} için grafik hazırlanıyor...")

        try:
            # Tahmin
            y_pred = model.predict(X_test)

            # Matris Hesaplama
            cm = confusion_matrix(y_test, y_pred)

            # Normalizasyon (0'a bölme hatasını yutmak için errstate kullanılır)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm)  # NaN varsa 0 yap

            # Çizim (Seaborn Heatmap)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=classes, yticklabels=classes,
                        annot_kws={"size": 12}, cbar=True)

            # Başlık ve Etiketler
            plt.title(f'Confusion Matrix (Normalized)\n{model_name}', fontsize=14, fontweight='bold', pad=15)
            plt.ylabel('Gerçek Sınıf (True)', fontsize=12)
            plt.xlabel('Tahmin Edilen Sınıf (Predicted)', fontsize=12)

            plt.tight_layout()

            # Kaydetme (Dosya ismindeki boşlukları alt çizgi yapar)
            safe_name = model_name.replace(" ", "_")
            save_path = f"{output_folder}/CM_{safe_name}.png"
            plt.savefig(save_path, dpi=300)
            plt.close()  # Belleği şişirmemek için grafiği kapat

            print(f" Kaydedildi: {save_path}")

        except Exception as e:
            print(f" {model_name} çizilirken hata oluştu: {e}")

    print("\n--- Tüm Confusion Matrix çizimleri tamamlandı! ---")



# Execution
file_path = 'izmirim-kart-ulasim-istatistikleri-guncel-extended.csv'
if os.path.exists(file_path):
    df_raw = prepare_data(file_path)
    df_processed = feature_engineering(df_raw)
    train_and_compare_models_time_series(df_processed)
else:
    print("File not found.")