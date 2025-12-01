from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import utils.utils as utils
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
        accuracies.append(inference.test(fold=i, k=k)[0])
    
    print(f"Average accuracy: {sum(accuracies) / k:.2%}")


# 5-fold cross-validation function with quantile-based discretization (100 buckets)
def second_iteration():
    preprocessing.BUCKETS = 100
    preprocessing.parse_data(quantile_based=True)
    preprocessing.shuffle_data()

    learning.INPUT_CSV = "data/shuffle_data.csv"
    
    k = 5
    accuracies = []
    full_y_true, full_y_pred = [], []
    for i in range(k):
        learning.generate_cpts(fold=i, k=k)
        accuracy, y_true, y_pred = inference.test(fold=i, k=k)
        print(y_true.count(15))
        full_y_true.extend(y_true)
        full_y_pred.extend(y_pred)
        accuracies.append(accuracy)

    print(f"Average accuracy: {sum(accuracies) / k:.2%}")

    # Confusion matrix
    genres = utils.load("fma_metadata/genres.csv")
    labels = sorted(set(genres['top_level']))
    genre_map = dict(zip(genres.index, genres['title']))
    display_labels = [genre_map[g] for g in labels]

    cm = confusion_matrix(full_y_true, full_y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.show()


def custom_inference():
    preprocessing.BUCKETS = 100

    learning.INPUT_CSV = "data/final_data.csv"
    learning.generate_cpts(fold=0, k=1)  # Train on all data



if __name__ == "__main__":
    # first_iteration()
    second_iteration()

    # custom_inference()