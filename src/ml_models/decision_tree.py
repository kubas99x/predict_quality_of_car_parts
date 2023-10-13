import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from mlflow import log_params, log_metrics, start_run
import mlflow
import mlflow.keras


def decision_tree_model(x_train, x_valid, x_test, y_train, y_valid, y_test, max_depth_=10, run_name_='standard_run', comment='no comment'):

    artifact_directory = "decision_tree"
    mlflow.set_experiment(artifact_directory)
    mlflow.sklearn.autolog()

    with start_run(run_name=run_name_):

        clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth_)
        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)
        predicted_classes = (predictions > 0.5).astype(int)  
        recall_nok = recall_score(y_test, predicted_classes, pos_label=1)
        recall_ok = recall_score(y_test, predicted_classes, pos_label=0)
        accuracy = accuracy_score(y_test, predicted_classes)

        log_params({'comment': comment, 'used_columns_shape':x_train.shape})
        log_metrics({'recall_nok':recall_nok, 'recall_ok':recall_ok, 'acc_test':accuracy})

        fig = plt.figure(figsize=(20 + (max_depth_/100) * 120, 20))

        plot_tree(clf,
              feature_names=list(x_train.columns),
              class_names=['ok', 'nok'],
              filled=True,
              rounded=True)
        plt.savefig('decision_tree_graph.png')
        mlflow.log_artifact("decision_tree_graph.png")

    return clf, max_depth_

def print_stats(clf, x_test, y_test):
    y_pred = clf.predict(x_test)

    # tworzenie macierzy konfuzji
    cm = confusion_matrix(y_test, y_pred)
    # oblicznie dokładności
    acc = accuracy_score(y_test, y_pred)
    
    # tworzenie wizualizacji macierzy konfuzji
    cm = cm[::-1]
    cm = pd.DataFrame(data=cm, columns=['pred_0', 'pred_1'], index=['true_1', 'true_0'])

    fig = ff.create_annotated_heatmap(
        z=cm.values, 
        x=list(cm.columns), 
        y=list(cm.index),
        colorscale='ice',
        showscale=True,
        reversescale=True)
    
    fig.update_layout(
        width=500,
        height=500,
        title=f'Confusion Matrix: accuracy = {acc * 100}%',
        font_size=16)

    fig.show()

    # wyświetlanie raportu modelu
    print(classification_report(y_test, y_pred))




