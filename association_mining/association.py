import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CSV_PATH = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"

df = pd.read_csv(
    CSV_PATH,
    sep=";",
    engine="python",
    on_bad_lines="skip"
)


usage_cols = [
    "FULL_FARE",
    "STUDENT",
    "TEACHER",
    "SIXTY_YEARS_OLD",
    "CHILD",
    "PERSONNEL",
    "FREE",
    "TICKET",
    "BANK CARD"
]

usage_cols = [c for c in usage_cols if c in df.columns]

print("Usage columns:", usage_cols)


context_cols = [
    "SEASON",
    "WEEKDAY",
    "DAY_TYPE",
    "IS_HOLIDAY",
    "IS_SCHOOL_OPEN",
    "IS_EVE"
]

context_cols = [c for c in context_cols if c in df.columns]

print("Context columns:", context_cols)


df = df[usage_cols + context_cols].copy()


for col in usage_cols:
    median_val = df[col].median()
    df[f"High_{col}"] = df[col] > median_val
    df[f"Low_{col}"] = df[col] <= median_val


binary_context = [
    "IS_HOLIDAY",
    "IS_SCHOOL_OPEN",
    "IS_EVE"
]

for col in binary_context:
    if col in df.columns:
        df[col] = df[col] == 1


categorical_context = ["SEASON", "DAY_TYPE"]

for col in categorical_context:
    if col in df.columns:
        for val in df[col].dropna().unique():
            df[f"{col}_{val}"] = df[col] == val



item_cols = []

item_cols += [c for c in df.columns if c.startswith("High_")]
item_cols += [c for c in df.columns if c.startswith("Low_")]

item_cols += [c for c in binary_context if c in df.columns]

item_cols += [c for c in df.columns if c.startswith("SEASON_")]
item_cols += [c for c in df.columns if c.startswith("DAY_TYPE_")]

basket = df[item_cols].astype(bool)

print("Basket shape:", basket.shape)

import numpy as np
np.seterr(divide="ignore", invalid="ignore")

frequent_itemsets = apriori(
    basket,
    min_support=0.02,
    use_colnames=True
)

print("Frequent itemsets count:", frequent_itemsets.shape[0])

if frequent_itemsets.empty:
    raise ValueError("No frequent itemsets found. Lower min_support.")



rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

def remove_redundant_rules(rules_df):
    """
    Remove redundant rules where a rule with a superset antecedent
    does not significantly improve confidence compared to a simpler rule.
    """
    cleaned_rules = []
    for idx, rule in rules_df.iterrows():
        redundant = False
        for _, other in rules_df.iterrows():
            if (
                rule["consequents"] == other["consequents"]
                and set(other["antecedents"]).issubset(set(rule["antecedents"]))
                and other["confidence"] >= rule["confidence"]
                and set(other["antecedents"]) != set(rule["antecedents"])
            ):
                redundant = True
                break
        if not redundant:
            cleaned_rules.append(rule)
    return pd.DataFrame(cleaned_rules)


rules = rules[
    (rules["antecedents"].apply(len).isin([1, 2])) &
    (rules["consequents"].apply(len) == 1) &
    (rules["support"] >= 0.02)
]

rules = remove_redundant_rules(rules)

interesting_rules = rules[
    (rules["lift"] >= 1.5) &
    (rules["lift"] <= 5)
].sort_values(["lift", "confidence"], ascending=False)

final_rules = interesting_rules[
    ["antecedents", "consequents", "support", "confidence", "lift"]
].head(30)

print("\n=== FINAL ASSOCIATION RULES ===")
print(final_rules)


# ===============================
# CONTEXTUAL ANALYSIS
# ===============================

# Automatically selected contextual dimensions based on dataset
# 1) School Open vs Closed
# 2) Holiday vs Non-Holiday

CONTEXT_CASES = []

if "IS_SCHOOL_OPEN" in df.columns:
    CONTEXT_CASES.append(
        ("IS_SCHOOL_OPEN", True, "SCHOOL OPEN")
    )
    CONTEXT_CASES.append(
        ("IS_SCHOOL_OPEN", False, "SCHOOL CLOSED")
    )

if "IS_HOLIDAY" in df.columns:
    CONTEXT_CASES.append(
        ("IS_HOLIDAY", True, "HOLIDAY")
    )
    CONTEXT_CASES.append(
        ("IS_HOLIDAY", False, "NON-HOLIDAY")
    )

if "SEASON" in df.columns:
    for season in df["SEASON"].dropna().unique():
        CONTEXT_CASES.append(
            (f"SEASON_{season}", True, f"SEASON: {season.upper()}")
        )

def mine_contextual_rules(context_column, context_value, label):
    if context_column not in df.columns:
        return None
    subset = df[df[context_column] == context_value]
    if subset.empty:
        return None

    basket_ctx = subset[item_cols].astype(bool)

    fi = apriori(
        basket_ctx,
        min_support=0.02,
        use_colnames=True
    )

    if fi.empty:
        return None

    import numpy as np
    np.seterr(divide="ignore", invalid="ignore")

    ctx_rules = association_rules(
        fi,
        metric="confidence",
        min_threshold=0.6
    )

    ctx_rules = ctx_rules[
        (ctx_rules["antecedents"].apply(len).isin([1, 2])) &
        (ctx_rules["consequents"].apply(len) == 1)
    ].sort_values("lift", ascending=False)

    print(f"\n=== CONTEXTUAL RULES: {label} ===")
    print(ctx_rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))


# ===============================
# RUN DATASET-AWARE CONTEXTUAL ANALYSIS
# ===============================

for context_column, context_value, label in CONTEXT_CASES:
    mine_contextual_rules(context_column, context_value, label)
