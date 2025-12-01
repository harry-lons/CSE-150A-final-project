import numpy as np
import pandas as pd
import json

import utils.utils as utils

BUCKETS = 100

# tracks.csv has a lot of miscellaneous info we don't care about for our project
def replace_tracks(tracks_file='fma_metadata/tracks.csv'):
    tracks = utils.load(tracks_file)

    # Drop irrelevant features
    tracks = tracks['track'].drop(
        columns=['bit_rate', 'comments', 'composer', 'date_recorded',
                 'duration', 'favorites', 'genres', 'genres_all',
                 'information', 'interest', 'language_code', 'license',
                 'listens', 'lyricist', 'number', 'publisher', 'tags']
    )

    tracks.to_csv('data/short_tracks.csv', index=False)


# features.csv needs to be shortened to get rid of redundant/lengthy variables
def replace_features(features_file='fma_metadata/features.csv', custom_features=None):
    if custom_features is None:
        features = utils.load(features_file)
    else:
        features = custom_features.copy()

    # Drop irrelevant features
    features = features.drop(
        columns=['chroma_cens', 'chroma_cqt', 'spectral_rolloff'],
        level=0
    )

    # Only keep mean and std functions for each feature
    features = features.loc[
        :, 
        features.columns.get_level_values(1).isin(['mean', 'std'])
    ]

    # Keep first 5 mfcc, chroma_stft, spectral_contrast features
    features = features.loc[
        :,
        ~(
            features.columns.get_level_values(0).isin([
                'mfcc',
                'chroma_stft',
                'spectral_contrast'
            ])
            &
            ~features.columns.get_level_values(2).isin(['01','02','03','04','05'])
        )
    ]

    # Collapse all of the levels
    features.columns = [
        f"{feat}_{stat}_{chan}"
        for feat, stat, chan in features.columns
    ]

    if custom_features is None:
        features.to_csv('data/short_features.csv', index=False)
    else:
        return features


# Discretizes tracks and features and merges them together
def parse_data(tracks_file='data/short_tracks.csv',
               features_file='data/short_features.csv',
               quantile_based=True):
    tracks = pd.read_csv(tracks_file)
    features = pd.read_csv(features_file)

    # Compute min and max per column
    mins = features.min()
    maxs = features.max()

    # Apply discretizer to each column
    discrete_features = features.copy()
    bin_edges = {}
    for col in features.columns:
        if quantile_based:
            discrete_features[col], bins = discretizer_quantile(features[col], BUCKETS)

            # NOTE: Custom inference is only available if data uses quantile-based discretization
            bin_edges[col] = bins.tolist()

        else:
            discrete_features[col] = features[col].apply(
                lambda x: discretizer(x, mins[col], maxs[col], BUCKETS)
            )

    # Save bin edges for custom inference
    json.dump(bin_edges, open('data/bin_edges.json', 'w'))

    # Merge tracks and features together
    final_data = pd.concat([tracks, discrete_features], axis=1)

    # Get rid of tracks where genre_top is not populated
    final_data = final_data.dropna(subset=['genre_top'])
    
    # Remap genre_top titles to use genre_id specified in genres.csv
    genres = utils.load('fma_metadata/genres.csv')
    genre_map = dict(zip(genres['title'], genres['top_level']))
    final_data['genre_id'] = final_data['genre_top'].map(genre_map)

    final_data.to_csv('data/final_data.csv', index=False)


# Used to discretize new songs for custom inference
def custom_discretizer(features, bin_edges, buckets):
    def discretize_new_value(value, bins, buckets):
        idx = np.digitize([value], bins[1:-1])[0]
        return min(idx, buckets-1)
    
    discrete_features = features.copy()
    for col in features.columns:
        bins = bin_edges[col]
        discrete_features[col] = features[col].apply(
            lambda x: discretize_new_value(x, bins, buckets)
        )

    return discrete_features

# Shuffle final_data.csv
def shuffle_data(data_file='data/final_data.csv', random_state=815):
    df = pd.read_csv(data_file)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df.to_csv('data/shuffle_data.csv', index=False)


# Convert variable over a continuous range into a discrete value
def discretizer(value, minimum, maximum, buckets):
    if value < minimum: return 0
    elif value >= maximum: return buckets - 1

    bucket_size = (maximum - minimum) / buckets

    return int((value - minimum) / bucket_size)

# Quantile-based discretization
def discretizer_quantile(col, buckets):
    categories, bins = pd.qcut(
        col, q=buckets, labels=False, retbins=True, duplicates='drop'
    )
    return categories, bins

if __name__ == "__main__":
    # replace_tracks()
    # replace_features()
    parse_data()