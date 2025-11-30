import numpy as np
import pandas as pd

BUCKETS = 10

def infer(CPTs, x: np.ndarray):
    """
    Infer the genre of a given song using the CPTs.

    x: one feature vector of length num_features to infer G from
    """
    cpts_array, priors_array, sorted_genres, features = CPTs

    if len(x) != len(features):
        raise ValueError(f"Expected {len(features)} features, but got {len(x)}")

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-9
    
    # Initialize with log priors: log(P(G))
    log_probs = np.log(priors_array + epsilon)
    
    # Add log likelihoods: sum(log(P(xi | G)))
    # cpts_array shape is (num_features, num_genres, BUCKETS)
    for i, val in enumerate(x):
        val = int(val)
        if 0 <= val < BUCKETS:
            # Get probabilities for this feature value across all genres
            probs = cpts_array[i, :, val]
            log_probs += np.log(probs + epsilon)
            
    # Find the index of the genre with the maximum posterior probability
    best_genre_idx = np.argmax(log_probs)
    
    return sorted_genres[best_genre_idx]

def loadCPTs(filename="data/cpt_output.txt"):
    """
    Parses the CPT file and returns:
      - cpts_array: (num_features, num_genres, BUCKETS)
      - priors_array: (num_genres,)
      - genres: list of genre IDs
      - features: list of feature names
    """
    priors = {}
    cpts_data = {} # feature -> {genre -> [probs]}
    current_feature = None
    mode = None # "PG" or "Feat"

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("==="):
                continue

            if line == "P(G):":
                mode = "PG"
                continue
            
            if line.startswith("P(") and "| G):" in line:
                # Parse feature name from "P(feature_name | G):"
                current_feature = line[2:].split(" | G):")[0]
                cpts_data[current_feature] = {}
                mode = "Feat"
                continue

            if mode == "PG":
                parts = line.split()
                if len(parts) == 2:
                    gid = int(parts[0])
                    prob = float(parts[1])
                    priors[gid] = prob
            
            elif mode == "Feat":
                parts = line.split()
                if len(parts) == 3:
                    gid = int(parts[0])
                    val = int(parts[1])
                    prob = float(parts[2])
                    
                    if gid not in cpts_data[current_feature]:
                        cpts_data[current_feature][gid] = np.zeros(BUCKETS)
                    
                    if 0 <= val < BUCKETS:
                        cpts_data[current_feature][gid][val] = prob

    # Process into numpy arrays
    sorted_genres = sorted(priors.keys())
    num_genres = len(sorted_genres)
    
    priors_array = np.array([priors[g] for g in sorted_genres])

    # Assume features are encountered in order in the file
    features = list(cpts_data.keys())
    num_features = len(features)

    # Shape: (num_features, num_genres, BUCEKTS options)
    cpts_array = np.zeros((num_features, num_genres, BUCKETS))

    for f_idx, feat in enumerate(features):
        for g_idx, genre in enumerate(sorted_genres):
            if genre in cpts_data[feat]:
                cpts_array[f_idx, g_idx, :] = cpts_data[feat][genre]
            else:
                # isseu reading data
                pass

    return cpts_array, priors_array, sorted_genres, features




def test(cpts_file="data/cpt_output.txt", data_file="data/final_data.csv"):
    """
    Tests inference on the last 20% of the data.
    """
    print("Loading CPTs...")
    cpts_args = loadCPTs(cpts_file)
    # unpack to get feature names
    _, _, _, features = cpts_args
    
    print("Loading data...")
    df = pd.read_csv(data_file)
    
    # Split data: use last 20% for testing
    cutoff = int(len(df) * 0.8)
    test_df = df.iloc[cutoff:].copy()
    print(f"Testing on {len(test_df)} samples (last 20% of data)")
    
    # Clean data (same as in learning.py)
    # Ensure we only look at the features the model knows about
    test_df[features] = test_df[features].replace([np.inf, -np.inf], np.nan)
    test_df[features] = test_df[features].fillna(0)
    test_df[features] = test_df[features].astype(int)
    
    correct = 0
    total = len(test_df)
    
    for idx, row in test_df.iterrows():
        ground_truth = row['genre_id']
        x = row[features].values
        
        predicted_genre = infer(cpts_args, x)
        
        if predicted_genre == ground_truth:
            correct += 1
            
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    test()