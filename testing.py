import matplotlib.pyplot as plt

import preprocessing
import learning
import inference


# 5-fold cross-validation function with min/max discretization (10 buckets)
def first_iteration():
    preprocessing.BUCKETS = 10
    preprocessing.parse_data(quantile_based=False)
    preprocessing.shuffle_data()

    learning.INPUT_CSV = "data/shuffle_data.csv"
    
    k = 5
    accuracies = []
    for i in range(k):
        learning.generate_cpts(fold=i, k=k)
        accuracies.append(inference.test(fold=i, k=k))
    
    print(f"Average accuracy: {sum(accuracies) / k:.2%}")


# 5-fold cross-validation function with quantile-based discretization (100 buckets)
def second_iteration():
    preprocessing.BUCKETS = 100
    preprocessing.parse_data(quantile_based=True)
    preprocessing.shuffle_data()

    learning.INPUT_CSV = "data/shuffle_data.csv"
    
    k = 5
    accuracies = []
    for i in range(k):
        learning.generate_cpts(fold=i, k=k)
        accuracies.append(inference.test(fold=i, k=k))
    
    print(f"Average accuracy: {sum(accuracies) / k:.2%}")


def custom_inference():
    preprocessing.BUCKETS = 100

    learning.INPUT_CSV = "data/final_data.csv"
    learning.generate_cpts(fold=0, k=1)  # Train on all data



if __name__ == "__main__":
    first_iteration()
    second_iteration()

    custom_inference()