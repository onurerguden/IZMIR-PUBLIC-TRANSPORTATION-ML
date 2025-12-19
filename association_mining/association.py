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
    "IS_EXAM",
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
    "IS_EXAM",
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
rules = rules[
    (rules["antecedents"].apply(len) <= 2) &
    (rules["consequents"].apply(len) == 1)
    ]

interesting_rules = rules[
    (rules["lift"] >= 1.5) &
    (rules["lift"] <= 5) &
    (rules["support"] >= 0.02)
    ].sort_values("lift", ascending=False)

interesting_rules = interesting_rules[
    interesting_rules["antecedents"].astype(str) <
    interesting_rules["consequents"].astype(str)
    ]


final_rules = interesting_rules[
    ["antecedents", "consequents", "support", "confidence", "lift"]
].head(10)

print("\n=== FINAL ASSOCIATION RULES ===")
print(final_rules)
