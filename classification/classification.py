import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler, LabelEncoder

# =======================
# Configuration
# =======================
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (10, 6),
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def prepare_data(file_path):
    """Load and prepare raw data"""
    logger.info("=== DATA LOADING PHASE ===")
    df = pd.read_csv(file_path, sep=";")
    logger.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    try:
        df["DATE"] = pd.to_datetime(df["DATE"], format='%d.%m.%Y')
    except:
        df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

    passenger_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD',
                      'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']

    for col in passenger_cols:
        if col not in df.columns:
            df[col] = 0

    df['TOTAL_PASSENGERS'] = df[passenger_cols].sum(axis=1)
    df = df.sort_values(['INSTITUTION', 'DATE']).reset_index(drop=True)

    logger.info(f"Date range: {df['DATE'].min().date()} to {df['DATE'].max().date()}")
    logger.info(f"Passenger stats - Mean: {df['TOTAL_PASSENGERS'].mean():.0f}, "
                f"Std: {df['TOTAL_PASSENGERS'].std():.0f}")

    return df


def feature_engineering(df, target_col='TOTAL_PASSENGERS'):
    """
    Create time series features WITHOUT data leakage
    CRITICAL: All features must use ONLY past information
    """
    logger.info("=== FEATURE ENGINEERING PHASE ===")

    # Calendar features (safe - no leakage)
    df['MONTH'] = df['DATE'].dt.month
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
    df['DAY_OF_MONTH'] = df['DATE'].dt.day

    # Cyclical encoding (safe)
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)

    # CRITICAL FIX: Lag features must use shift(1) BEFORE any calculation
    # This ensures we NEVER see today's value
    df['LAG_1'] = df.groupby('INSTITUTION')[target_col].shift(1)
    df['LAG_2'] = df.groupby('INSTITUTION')[target_col].shift(2)
    df['LAG_7'] = df.groupby('INSTITUTION')[target_col].shift(7)
    df['LAG_14'] = df.groupby('INSTITUTION')[target_col].shift(14)

    # CRITICAL FIX: Rolling windows must be calculated on ALREADY SHIFTED data
    # This prevents the rolling window from including today's value
    for window in [3, 7, 14]:
        col_name = f'ROLL_MEAN_{window}'
        df[col_name] = (df.groupby('INSTITUTION')['LAG_1']
                        .transform(lambda x: x.rolling(window, min_periods=max(2, window // 2)).mean()))

        col_name_std = f'ROLL_STD_{window}'
        df[col_name_std] = (df.groupby('INSTITUTION')['LAG_1']
                            .transform(lambda x: x.rolling(window, min_periods=max(2, window // 2)).std()))

    # Day-of-week average (historical pattern)
    df['DOW_MEAN'] = (df.groupby(['INSTITUTION', 'DAY_OF_WEEK'])['LAG_1']
                      .transform(lambda x: x.expanding(min_periods=2).mean()))

    logger.info(f"Features created. Total columns: {len(df.columns)}")

    # Verify no leakage: Check if any LAG correlates too highly with target
    lag_cols = [c for c in df.columns if c.startswith('LAG_') or c.startswith('ROLL_')]
    if len(lag_cols) > 0:
        corr_check = df[lag_cols + [target_col]].corr()[target_col].drop(target_col)
        max_corr = corr_check.abs().max()
        logger.info(f"Max correlation between lags and target: {max_corr:.3f}")
        if max_corr > 0.99:
            logger.warning("  POTENTIAL LEAKAGE: Correlation > 0.99 detected!")

    return df


def create_target_variable(df, train_split_date):
    """
    Create classification target based on TRAIN data only
    CRITICAL FIX: Use percentiles from train data for better generalization
    """
    logger.info("=== TARGET VARIABLE CREATION ===")

    # Calculate thresholds ONLY from training data
    train_data = df[df['DATE'] <= train_split_date]['TOTAL_PASSENGERS']

    # Use 25th and 75th percentiles for more robust thresholds
    q1 = train_data.quantile(0.25)
    q2 = train_data.quantile(0.75)

    logger.info(f"Classification thresholds (from TRAIN only):")
    logger.info(f"  LOW <= {q1:.0f} (25th percentile)")
    logger.info(f"  MEDIUM: {q1:.0f} - {q2:.0f}")
    logger.info(f"  HIGH > {q2:.0f} (75th percentile)")

    def get_class(x):
        if x <= q1:
            return 'LOW'
        elif x <= q2:
            return 'MEDIUM'
        else:
            return 'HIGH'

    # Apply classification to all data
    df['PASSENGER_LEVEL'] = df['TOTAL_PASSENGERS'].apply(get_class)

    # CRITICAL: Shift target to predict NEXT day (t+1)
    df['TARGET'] = df.groupby('INSTITUTION')['PASSENGER_LEVEL'].shift(-1)

    return df, q1, q2


def train_and_evaluate_models(df):
    """Train and evaluate models with proper time series split"""
    logger.info("=== MODEL TRAINING AND EVALUATION ===")

    # Time series split (70/30 for more test data)
    split_date = df['DATE'].quantile(0.70)

    # Create target variable using only train data
    df, q1, q2 = create_target_variable(df, split_date)

    # Split data
    train_df = df[df['DATE'] <= split_date].copy()
    test_df = df[df['DATE'] > split_date].copy()

    logger.info(f"Split date: {split_date.date()}")
    logger.info(f"Train set: {len(train_df)} rows ({train_df['DATE'].min().date()} to {train_df['DATE'].max().date()})")
    logger.info(f"Test set: {len(test_df)} rows ({test_df['DATE'].min().date()} to {test_df['DATE'].max().date()})")

    # Remove NaN values
    feature_cols = [c for c in df.columns if c.startswith('LAG_') or c.startswith('ROLL_') or c.startswith('DOW_')]
    feature_cols.append('TARGET')

    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)

    logger.info(f"After NaN removal - Train: {len(train_df)}, Test: {len(test_df)}")

    # Class distribution
    logger.info("\nTrain class distribution:")
    train_dist = train_df['TARGET'].value_counts(normalize=True).sort_index()
    for cls, pct in train_dist.items():
        logger.info(f"  {cls}: {pct:.3f}")

    logger.info("\nTest class distribution:")
    test_dist = test_df['TARGET'].value_counts(normalize=True).sort_index()
    for cls, pct in test_dist.items():
        logger.info(f"  {cls}: {pct:.3f}")

    # Calculate class distribution difference (drift indicator)
    drift = np.abs(train_dist.values - test_dist.values).mean()
    logger.info(f"\n Temporal drift (avg class shift): {drift:.3f}")
    if drift > 0.1:
        logger.warning("High temporal drift detected - model may not generalize well")

    # Feature preparation
    categorical_cols = ['INSTITUTION', 'IS_HOLIDAY']
    numerical_cols = (['MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS', 'IS_WEEKEND', 'DAY_OF_MONTH'] +
                      [c for c in train_df.columns if
                       c.startswith('LAG_') or c.startswith('ROLL_') or c.startswith('DOW_')])

    X_train = pd.get_dummies(train_df[categorical_cols + numerical_cols], drop_first=True)
    X_test = pd.get_dummies(test_df[categorical_cols + numerical_cols], drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['TARGET'])
    y_test = le.transform(test_df['TARGET'])

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"\nFeature matrix shape: {X_train_scaled.shape}")
    logger.info(f"Classes: {le.classes_}")

    # CRITICAL: MORE AGGRESSIVE REGULARIZATION to prevent overfitting
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,  # Very shallow
            min_samples_split=200,  # High minimum
            min_samples_leaf=100,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,  # Fewer trees
            max_depth=6,  # Shallow trees
            min_samples_split=200,
            min_samples_leaf=100,
            class_weight='balanced',
            max_features='sqrt',
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=100),  # Very weak learners
            n_estimators=30,
            learning_rate=0.5,
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf',
            C=0.5,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(
            n_neighbors=50,  # More neighbors
            weights='distance'
        )
    }

    results = {}
    predictions = {}
    all_reports = {}

    print("\n" + "=" * 90)
    print(f"{'MODEL':<20} | {'ACCURACY':<10} | {'F1-MACRO':<10} | {'F1-LOW':<8} | {'F1-MED':<8} | {'F1-HIGH':<8}")
    print("-" * 90)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        results[name] = {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class
        }
        predictions[name] = y_pred

        # Store classification report
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        all_reports[name] = report

        print(f"{name:<20} | {acc * 100:>9.2f}% | {f1_macro * 100:>9.2f}% | "
              f"{f1_per_class[0] * 100:>7.1f}% | {f1_per_class[2] * 100:>7.1f}% | {f1_per_class[1] * 100:>7.1f}%")

    print("=" * 90)

    # Generate plots
    plot_model_comparison(results)
    plot_ensemble_comparison(results)
    plot_confusion_matrices(models, X_test_scaled, y_test, le, predictions)

    # Feature importance analysis
    analyze_feature_importance(models, X_train.columns)

    return results, models, X_train.columns


def analyze_feature_importance(models, feature_names):
    """Analyze and log feature importance for tree-based models"""
    logger.info("\n=== FEATURE IMPORTANCE ANALYSIS ===")

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-5:][::-1]  # Top 5

            logger.info(f"\nTop 5 features for {name}:")
            for idx in indices:
                logger.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")


def plot_model_comparison(results):
    """Plot comparison of all models"""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(model_names)))
    bars = ax.bar(model_names, accuracies, color=colors, edgecolor='navy', linewidth=1.5)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height * 100:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: model_comparison.png")


def plot_ensemble_comparison(results):
    """Plot comparison of tree-based ensemble methods only"""
    ensemble_models = ['Decision Tree', 'Random Forest', 'AdaBoost']

    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = [results[m]['accuracy'] for m in ensemble_models]
    f1_scores = [results[m]['f1_macro'] for m in ensemble_models]

    x = np.arange(len(ensemble_models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, accuracies, width, label='Accuracy',
                   color='steelblue', edgecolor='navy', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, f1_scores, width, label='F1-Macro',
                   color='lightblue', edgecolor='navy', linewidth=1.5)

    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(ensemble_models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height * 100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("ensemble_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: ensemble_comparison.png")


def plot_confusion_matrices(models_dict, X_test, y_test, label_encoder, predictions):
    """Generate confusion matrices for all models"""
    if not os.path.exists('confusion_matrices'):
        os.makedirs('confusion_matrices')

    classes = label_encoder.classes_

    for name, model in models_dict.items():
        y_pred = predictions[name]
        cm = confusion_matrix(y_test, y_pred, normalize='true')

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Rate'}, linewidths=0.5, ax=ax)

        ax.set_ylabel('Actual Class')
        ax.set_xlabel('Predicted Class')

        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.tight_layout()
        plt.savefig(f"confusion_matrices/CM_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved {len(models_dict)} confusion matrices to confusion_matrices/")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    file_path = 'izmirim-kart-ulasim-istatistikleri-guncel-extended.csv'

    if os.path.exists(file_path):
        logger.info("Starting pipeline with STRICT anti-leakage measures...")
        logger.info("Expected: Accuracy 60-75% (realistic for time series prediction)\n")

        df_raw = prepare_data(file_path)
        df_processed = feature_engineering(df_raw)
        results, models, feature_names = train_and_evaluate_models(df_processed)

        logger.info("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info("If accuracy is still >90%, check for additional leakage sources")
    else:
        logger.error(f"File not found: {file_path}")