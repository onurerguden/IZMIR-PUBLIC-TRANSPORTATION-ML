import pandas as pd


def load_data():
    df = pd.read_csv("data/current-data/izmirim-kart-ulasim-istatistikleri-guncel-extended.csv", sep=";")
    if "_id" in df.columns:
        df.drop("_id", axis=1, inplace=True)
    print("Data loaded...")

    # Refactor: fix Turkish character corruption
    replacements = {
        'Ý': 'İ', 'ý': 'ı',
        'þ': 'ş', 'Þ': 'Ş',
        'ð': 'ğ', 'Ð': 'Ğ'
    }
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace(replacements, regex=True)

    print("Character encoding fixed in DataFrame...")
    return df
def show_statistics(df):
    print("size of the data: ",df.shape)
    print("number of missing values: ",df.isnull().sum().sum())
    print("mode of the categorical attribute INSTITUTION: ",df["INSTITUTION"].mode()[0])


