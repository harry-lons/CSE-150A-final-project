import matplotlib.pyplot as plt

import preprocessing
import learning
import inference


# 5-fold cross-validation function with min/max discretization (10 buckets)
def first_iteration():
    preprocessing.BUCKETS = 10
    preprocessing.parse_data(quantile_based=False)
    preprocessing.shuffle_data(random_state=616)

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
    preprocessing.shuffle_data(random_state=815)

    learning.INPUT_CSV = "data/shuffle_data.csv"
    
    k = 5
    accuracies = []
    for i in range(k):
        learning.generate_cpts(fold=i, k=k)
        accuracies.append(inference.test(fold=i, k=k))
    
    print(f"Average accuracy: {sum(accuracies) / k:.2%}")


if __name__ == "__main__":
    # first_iteration()
    second_iteration()