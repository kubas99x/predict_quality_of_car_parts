import mlflow.keras
import mlflow
import mlflow.tensorflow
import umap
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np
from sklearn.metrics import confusion_matrix
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

def distribution_of_probability_plot(predictions, y_test,show_figure=False):
    
    predictions = predictions.reshape(-1, 1)
    pred_class_test = {'ok': predictions[y_test == 0],
                       'nok': predictions[y_test == 1]}

    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (key, pred_probability) in enumerate(pred_class_test.items()):
        bins = np.arange(0, 1.1, 0.1)
        hist, edges = np.histogram(pred_probability, bins=bins, density=True)
        x_ticks_labels = [f'{i:.1f} - {(i + 0.1):.1f}' for i in edges[:-1]]

        bar_color = 'red' if key == 'nok' else 'green'
        bar_positions = np.arange(len(x_ticks_labels)) + i * width

        ax.bar(bar_positions, hist * 10, width, alpha=0.7, label=key, color=bar_color)

        # Add value labels on top of each bar (rotated vertically)
        for x, value in zip(bar_positions, hist * 10):
            ax.text(x + width / 2 - 0.1, value + 1, f'{value:.2f}%', ha='center', va='bottom', rotation='vertical')

    ax.set_xlabel('Probability Range')
    ax.set_ylabel('Percentage')
    ax.set_title('Distribution of Probability for Classes')
    ax.set_xticks(bar_positions - width / 2)
    ax.set_xticklabels(x_ticks_labels, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 50)  # Set y-axis limit to 100%
    
    if show_figure:
        plt.show()

    return fig



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

def create_confusion_matrix(y_true, y_pred):

    cmat = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(cmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(x=j, y=i,s=cmat[i, j], va='center', ha='center', size='xx-large')
    
    ax.set_xlabel('Predictions', fontsize=18)
    ax.set_ylabel('Actuals', fontsize=18)
    ax.set_title('Confusion Matrix', fontsize=18)

    return fig
