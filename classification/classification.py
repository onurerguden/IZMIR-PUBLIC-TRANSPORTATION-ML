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
        plt.title("Decision Tree Visualization - Passenger Level Classification", fontsize=16, fontweight='bold')
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


# Execution
file_path = 'izmirim-kart-ulasim-istatistikleri-guncel-extended.csv'
if os.path.exists(file_path):
    df_raw = prepare_data(file_path)
    df_processed = feature_engineering(df_raw)
    train_and_compare_models_optimized(df_processed)
else:
    print("File not found.")