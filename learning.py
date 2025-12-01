import numpy as np
import pandas as pd

import preprocessing

INPUT_CSV = "data/final_data.csv"
OUTPUT_TXT = "data/cpt_output.txt"

def normalize_counts(counts):
    total = counts.sum()
    if total > 0:
        return counts / total 
    else: 
        return 0

def generate_cpts(fold, k=5):
    df = pd.read_csv(INPUT_CSV)

    n = len(df)
    fold_size = n // k
    start = fold * fold_size
    end = (fold + 1) * fold_size if fold < k - 1 else n

    if k != 1:
        df = pd.concat([df.iloc[:start], df.iloc[end:]]).reset_index(drop=True)
    print(f"Training on {len(df)} samples (fold={fold})")

    genre_col = "genre_id"

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove("genre_id")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)

    df[feature_cols] = df[feature_cols].astype(int)

    # Compute P(G)
    genre_counts = df[genre_col].value_counts().sort_index()
    P_G = normalize_counts(genre_counts)

    # Compute P(X_i | G)
    CPTs = {}
    genres = sorted(df[genre_col].unique())

    for feat in feature_cols:
        CPTs[feat] = {}

        for g in genres:
            subset = df[df[genre_col] == g]

            counts = subset[feat].value_counts()

            bucket_counts = np.zeros(preprocessing.BUCKETS, dtype=float)

            for bucket, ct in counts.items():
                if isinstance(bucket, (np.integer, int)) and 0 <= bucket < preprocessing.BUCKETS:
                    bucket_counts[int(bucket)] = ct

            CPTs[feat][g] = normalize_counts(bucket_counts)

    with open(OUTPUT_TXT, "w") as f:

        f.write("=============================================\n")
        f.write("P(G):\n")
        f.write("=============================================\n")
        for g, p in P_G.items():
            f.write(f"{g} {p:.6f}\n")
        f.write("\n\n")

        for feat in feature_cols:
            f.write("=============================================\n")
            f.write(f"P({feat} | G):\n")
            f.write("=============================================\n")

            for g in genres:
                probs = CPTs[feat][g]
                for value, p in enumerate(probs):
                    f.write(f"{g} {value} {p:.6f}\n")

            f.write("\n")

if __name__ == "__main__":
    generate_cpts(fold=4)
