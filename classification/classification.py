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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn import tree
import graphviz

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
    Create time series features with STRICT leakage prevention
    Strategy: Use ONLY aggregate historical patterns, avoid direct lags
    """
    logger.info("=== FEATURE ENGINEERING PHASE ===")

    # Calendar features (safe - no leakage)
    df['MONTH'] = df['DATE'].dt.month
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
    df['DAY_OF_MONTH'] = df['DATE'].dt.day
    df['WEEK_OF_YEAR'] = df['DATE'].dt.isocalendar().week

    # Cyclical encoding (safe)
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)

    # STRATEGY 1: Use distant lags only (reduces information leakage)
    df['LAG_7'] = df.groupby('INSTITUTION')[target_col].shift(7)
    df['LAG_14'] = df.groupby('INSTITUTION')[target_col].shift(14)
    df['LAG_21'] = df.groupby('INSTITUTION')[target_col].shift(21)
    df['LAG_28'] = df.groupby('INSTITUTION')[target_col].shift(28)

    # STRATEGY 2: Use expanding windows (cumulative averages, less sensitive)
    for lag_start in [7, 14]:
        col_name = f'EXPANDING_MEAN_{lag_start}'
        df[col_name] = (df.groupby('INSTITUTION')[target_col]
                        .shift(lag_start)
                        .expanding(min_periods=7)
                        .mean()
                        .reset_index(level=0, drop=True))

    # STRATEGY 3: Day-of-week historical pattern (excludes recent weeks)
    df['DOW_HIST_MEAN'] = (df.groupby(['INSTITUTION', 'DAY_OF_WEEK'])[target_col]
                           .transform(lambda x: x.shift(14).expanding(min_periods=4).mean()))

    # STRATEGY 4: Relative change features (less absolute value dependency)
    df['LAG_7_DIFF'] = df.groupby('INSTITUTION')[target_col].shift(7).diff()
    df['LAG_14_DIFF'] = df.groupby('INSTITUTION')[target_col].shift(14).diff()

    logger.info(f"Features created. Total columns: {len(df.columns)}")

    # Leakage check with WARNING threshold
    lag_cols = [c for c in df.columns if 'LAG' in c or 'EXPANDING' in c or 'DOW_HIST' in c]
    if len(lag_cols) > 0:
        corr_check = df[lag_cols + [target_col]].corr()[target_col].drop(target_col)
        max_corr = corr_check.abs().max()
        max_feature = corr_check.abs().idxmax()
        logger.info(f"Max correlation: {max_corr:.3f} ({max_feature})")

        if max_corr > 0.95:
            logger.error(f"🚨 SEVERE LEAKAGE: {max_feature} has correlation {max_corr:.3f}")
        elif max_corr > 0.85:
            logger.warning(f"⚠️  POTENTIAL LEAKAGE: {max_feature} has correlation {max_corr:.3f}")
        else:
            logger.info("✅ Correlation check passed")

    return df


def create_target_variable(df, train_split_date):
    """
    Create classification target with FIXED thresholds strategy
    Use train data but ensure meaningful class separation
    """
    logger.info("=== TARGET VARIABLE CREATION ===")

    # Calculate thresholds from training data
    train_data = df[df['DATE'] <= train_split_date]['TOTAL_PASSENGERS']

    # Use IQR-based thresholds for robustness
    q25 = train_data.quantile(0.25)
    q50 = train_data.quantile(0.50)
    q75 = train_data.quantile(0.75)

    # Strategy: Use median-based split to avoid extreme class imbalance
    threshold_low = q25
    threshold_high = q75

    logger.info(f"Classification thresholds (train-based):")
    logger.info(f"  Q25: {q25:.0f}")
    logger.info(f"  Q50 (median): {q50:.0f}")
    logger.info(f"  Q75: {q75:.0f}")
    logger.info(f"\nUsing: LOW <= {threshold_low:.0f}, HIGH > {threshold_high:.0f}")

    def get_class(x):
        if x <= threshold_low:
            return 'LOW'
        elif x <= threshold_high:
            return 'MEDIUM'
        else:
            return 'HIGH'

    # Apply classification
    df['PASSENGER_LEVEL'] = df['TOTAL_PASSENGERS'].apply(get_class)

    # Shift to predict next day
    df['TARGET'] = df.groupby('INSTITUTION')['PASSENGER_LEVEL'].shift(-1)

    # Log train distribution
    train_dist = df[df['DATE'] <= train_split_date]['PASSENGER_LEVEL'].value_counts(normalize=True)
    logger.info(f"\nTrain passenger level distribution:")
    for cls in ['LOW', 'MEDIUM', 'HIGH']:
        if cls in train_dist:
            logger.info(f"  {cls}: {train_dist[cls]:.3f}")

    return df, threshold_low, threshold_high


def train_and_evaluate_models(df):
    """Train and evaluate models with balanced accuracy focus"""
    logger.info("=== MODEL TRAINING AND EVALUATION ===")

    # Time series split (75/25)
    split_date = df['DATE'].quantile(0.75)

    # Create target variable
    df, threshold_low, threshold_high = create_target_variable(df, split_date)

    # Split data
    train_df = df[df['DATE'] <= split_date].copy()
    test_df = df[df['DATE'] > split_date].copy()

    logger.info(f"\nSplit date: {split_date.date()}")
    logger.info(f"Train: {len(train_df)} rows ({train_df['DATE'].min().date()} to {train_df['DATE'].max().date()})")
    logger.info(f"Test: {len(test_df)} rows ({test_df['DATE'].min().date()} to {test_df['DATE'].max().date()})")

    # Remove NaN values
    feature_cols = [c for c in df.columns if any(x in c for x in ['LAG', 'EXPANDING', 'DOW_HIST', 'DIFF'])]
    feature_cols.append('TARGET')

    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)

    logger.info(f"After NaN removal - Train: {len(train_df)}, Test: {len(test_df)}")

    # Class distribution analysis
    logger.info("\n📊 CLASS DISTRIBUTION:")
    logger.info("Train:")
    train_dist = train_df['TARGET'].value_counts(normalize=True).sort_index()
    for cls, pct in train_dist.items():
        logger.info(f"  {cls}: {pct:.3f} ({(pct * len(train_df)):.0f} samples)")

    logger.info("\nTest:")
    test_dist = test_df['TARGET'].value_counts(normalize=True).sort_index()
    for cls, pct in test_dist.items():
        logger.info(f"  {cls}: {pct:.3f} ({(pct * len(test_df)):.0f} samples)")

    # Calculate temporal drift
    common_classes = set(train_dist.index) & set(test_dist.index)
    if common_classes:
        drift = np.mean([abs(train_dist.get(c, 0) - test_dist.get(c, 0)) for c in common_classes])
        logger.info(f"\n⚠️  Temporal drift: {drift:.3f}")
        if drift > 0.15:
            logger.warning("HIGH DRIFT: Consider resampling or different threshold strategy")

    # Feature preparation
    categorical_cols = ['INSTITUTION', 'IS_HOLIDAY']
    numerical_cols = (['MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS',
                       'IS_WEEKEND', 'DAY_OF_MONTH', 'WEEK_OF_YEAR'] +
                      [c for c in train_df.columns if any(x in c for x in ['LAG', 'EXPANDING', 'DOW_HIST', 'DIFF'])])

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

    logger.info(f"\nFeature matrix: {X_train_scaled.shape}")
    logger.info(f"Classes: {le.classes_}")

    # FINAL MODELS: Even more conservative to fight overfitting
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=300,
            min_samples_leaf=150,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=30,
            max_depth=5,
            min_samples_split=300,
            min_samples_leaf=150,
            class_weight='balanced',
            max_features='sqrt',
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=150),
            n_estimators=25,
            learning_rate=0.7,
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf',
            C=0.3,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(
            n_neighbors=60,
            weights='distance'
        )
    }

    results = {}
    predictions = {}

    print("\n" + "=" * 100)
    print(
        f"{'MODEL':<20} | {'ACCURACY':<10} | {'BAL-ACC':<10} | {'F1-MACRO':<10} | {'F1-LOW':<8} | {'F1-MED':<8} | {'F1-HIGH':<8}")
    print("-" * 100)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)  # Better metric for imbalanced classes
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])

        results[name] = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class
        }
        predictions[name] = y_pred

        print(f"{name:<20} | {acc * 100:>9.2f}% | {bal_acc * 100:>9.2f}% | {f1_macro * 100:>9.2f}% | "
              f"{f1_per_class[1] * 100:>7.1f}% | {f1_per_class[2] * 100:>7.1f}% | {f1_per_class[0] * 100:>7.1f}%")

    print("=" * 100)

    logger.info(f"\n💡 Use BALANCED ACCURACY for imbalanced test set comparison")

    # Generate plots
    plot_model_comparison(results)
    plot_ensemble_comparison(results)
    plot_confusion_matrices(models, X_test_scaled, y_test, le, predictions)
    analyze_feature_importance(models, X_train.columns)

    # Visualize Decision Tree
    visualize_decision_tree(models['Decision Tree'], X_train.columns, le.classes_)

    return results, models, X_train.columns


def analyze_feature_importance(models, feature_names):
    """Analyze and log feature importance for tree-based models"""
    logger.info("\n=== FEATURE IMPORTANCE ANALYSIS ===")

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Check if single feature dominates
            max_importance = importances.max()
            if max_importance > 0.7:
                logger.warning(f"⚠️  {name}: Single feature dominates ({max_importance:.3f})")

            indices = np.argsort(importances)[-5:][::-1]

            logger.info(f"\nTop 5 features for {name}:")
            for idx in indices:
                logger.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")


def plot_model_comparison(results):
    """Plot comparison of all models using balanced accuracy"""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(results.keys())
    accuracies = [results[m]['balanced_accuracy'] for m in model_names]

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(model_names)))
    bars = ax.bar(model_names, accuracies, color=colors, edgecolor='navy', linewidth=1.5)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Balanced Accuracy')
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
    """Plot comparison of tree-based ensemble methods"""
    ensemble_models = ['Decision Tree', 'Random Forest', 'AdaBoost']

    fig, ax = plt.subplots(figsize=(10, 6))

    bal_accs = [results[m]['balanced_accuracy'] for m in ensemble_models]
    f1_scores = [results[m]['f1_macro'] for m in ensemble_models]

    x = np.arange(len(ensemble_models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, bal_accs, width, label='Balanced Accuracy',
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


def visualize_decision_tree(dt_model, feature_names, class_names):
    """
    Visualize Decision Tree with best practices
    Creates both detailed and simplified versions
    """
    logger.info("\n=== DECISION TREE VISUALIZATION ===")

    # Method 1: High-quality matplotlib visualization
    fig, ax = plt.subplots(figsize=(20, 12))
    tree.plot_tree(dt_model,
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True,
                   rounded=True,
                   fontsize=9,
                   ax=ax,
                   proportion=True,
                   precision=2,
                   impurity=False)  # Hide impurity for cleaner look

    plt.tight_layout()
    plt.savefig("decision_tree_full.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: decision_tree_full.png (matplotlib version)")

    # Method 2: Simplified tree (top 2 levels only)
    fig, ax = plt.subplots(figsize=(16, 10))
    tree.plot_tree(dt_model,
                   max_depth=2,  # Show only top 2 levels
                   feature_names=feature_names,
                   class_names=class_names,
                   filled=True,
                   rounded=True,
                   fontsize=11,
                   ax=ax,
                   proportion=True,
                   precision=2)

    plt.tight_layout()
    plt.savefig("decision_tree_simplified.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: decision_tree_simplified.png (top 2 levels)")

    # Method 3: Graphviz DOT format (publication quality)
    try:
        dot_data = tree.export_graphviz(
            dt_model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
            proportion=True,
            precision=2,
            impurity=False
        )

        # Save DOT file
        with open('decision_tree.dot', 'w') as f:
            f.write(dot_data)
        logger.info("Saved: decision_tree.dot (Graphviz format)")

        # Try to render with Graphviz if available
        try:
            graph = graphviz.Source(dot_data)
            graph.render('decision_tree_graphviz', format='png', cleanup=True)
            logger.info("Saved: decision_tree_graphviz.png (Graphviz rendered)")
        except:
            logger.info("Graphviz not available for rendering (DOT file saved)")
    except Exception as e:
        logger.warning(f"Could not export Graphviz format: {e}")

    # Tree statistics
    logger.info(f"\nTree Statistics:")
    logger.info(f"  Max depth: {dt_model.get_depth()}")
    logger.info(f"  Number of leaves: {dt_model.get_n_leaves()}")
    logger.info(f"  Total nodes: {dt_model.tree_.node_count}")

    # Get most important split
    feature_importance = dt_model.feature_importances_
    top_feature_idx = feature_importance.argmax()
    logger.info(f"  Root split feature: {feature_names[top_feature_idx]}")
    logger.info(f"  Root feature importance: {feature_importance[top_feature_idx]:.4f}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    file_path = 'izmirim-kart-ulasim-istatistikleri-guncel-extended.csv'

    if os.path.exists(file_path):
        logger.info("=" * 80)
        logger.info("🚀 STRICT TIME SERIES CLASSIFICATION PIPELINE")
        logger.info("=" * 80)
        logger.info("Anti-leakage measures:")
        logger.info("  ✓ Distant lags only (7+ days)")
        logger.info("  ✓ Expanding windows (cumulative)")
        logger.info("  ✓ Conservative model parameters")
        logger.info("  ✓ Balanced accuracy metric")
        logger.info("=" * 80 + "\n")

        df_raw = prepare_data(file_path)
        df_processed = feature_engineering(df_raw)
        results, models, feature_names = train_and_evaluate_models(df_processed)

        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETED")
        logger.info("=" * 80)
    else:
        logger.error(f"File not found: {file_path}")