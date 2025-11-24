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
    full_data = pd.concat([tracks, features], axis=1)
    print(full_data)


# Convert variable over a continuous range into a discrete value
def discretizer(value, minimum, maximum, buckets):
    if value < minimum: return 0
    elif value >= maximum: return buckets - 1

    bucket_size = (maximum - minimum) / buckets

    return int((value - minimum) / bucket_size)

if __name__ == "__main__":
    load_data()