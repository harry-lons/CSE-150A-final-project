import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm

import utils

# tracks.csv has a lot of miscellaneous info we don't care about for our project
def replace_tracks():
    tracks = utils.load('fma_metadata/tracks.csv')
    tracks['track'].to_csv('fma_metadata/short_tracks.csv', index=False)


# Loads relevant data for performing MLE
def load_data():
    tracks = pd.read_csv('fma_metadata/short_tracks.csv')
    genres = utils.load('fma_metadata/genres.csv')
    features = utils.load('fma_metadata/features.csv')

    # NOTE: Originally we assumed that each audio feature had only one value associated with it. 
    # Variables like MFCC have a ton of values associated with it.

    # Also, delete rows where len(genres) == 0

    full_data = pd.concat([tracks, features], axis=1)
    
    return tracks


# Compute prior probabilities for each genre
def compute_prior(tracks):
    # NOTE: Technically we can use genres.csv for this... but since one song can have multiple genres
    # it won't be true probability values. Should talk about this later
    genre_counts = tracks['genres'].value_counts()
    prior_probs = genre_counts / len(tracks)

    return prior_probs 


# Convert variable over a continuous range into a discrete value
def discretizer(value, minimum, maximum, buckets):
    if value < minimum: return 0
    elif value >= maximum: return buckets - 1

    bucket_size = (maximum - minimum) / buckets

    return int((value - minimum) / bucket_size)


if __name__ == "__main__":
    tracks = load_data()

    compute_prior(tracks)