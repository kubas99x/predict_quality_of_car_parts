import mlflow.keras
import mlflow
import mlflow.tensorflow
import umap

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA

def distribution_of_probability(x_test, y_test, path_to_model = None, model_ = None):
    if path_to_model is not None:
        model = mlflow.keras.load_model(path_to_model)
    elif model_ is not None:
        model = model_
    else:
        return -1
    predictions = model.predict(x_test)
    range_table = np.arange(0, 1, 0.1)
    pred_class_test = {'ok' : predictions[y_test==0],
                       'nok' : predictions[y_test==1]}

    for key, pred_probability in pred_class_test.items():
        print(f"Procentowy rozkład prawdopodobienstwa dla klasy {key}")
        for i in range_table:
            if i != 0:
                percent_of_values = (np.sum((pred_probability > i) & (pred_probability <= i + 0.1)) / len(pred_probability)) * 100
            else:
                percent_of_values = (np.sum((pred_probability >= i) & (pred_probability <= i + 0.1)) / len(pred_probability)) * 100
            print(f'{i:.1f} - {(i+0.1):.1f} - {percent_of_values:.2f} %')


def umap_transformation(x_train_, x_valid_, x_test_, n_components_umap, umap_min_dist):

    x_train = np.copy(x_train_)
    x_valid = np.copy(x_valid_)
    x_test = np.copy(x_test_)

    umap_model = umap.UMAP(n_components=n_components_umap, min_dist=umap_min_dist)
    x_train = umap_model.fit_transform(x_train)
    x_valid = umap_model.transform(x_valid)
    x_test = umap_model.transform(x_test)

    return x_train, x_valid, x_test

def pca_transformation(x_train_, x_valid_, x_test_, n_components_):

    x_train = np.copy(x_train_)
    x_valid = np.copy(x_valid_)
    x_test = np.copy(x_test_)

    pca = PCA(n_components=n_components_)
    x_train = pca.fit_transform(x_train)
    x_valid = pca.transform(x_valid)
    x_test = pca.transform(x_test)

    return x_train, x_valid, x_test