import pandas as pd
import numpy as np
import graphviz
import plotly.figure_factory as ff
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow import log_params, log_metrics, start_run
import mlflow
import mlflow.keras
from sklearn.metrics import recall_score


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
        accuracy = accuracy_score(y_test, predictions)

        log_params({'comment': comment, 'used_columns_shape':x_train.shape})
        log_metrics({'recall_nok':recall_nok, 'recall_ok':recall_ok, 'acc_test':accuracy})

        # dot_data = tree.export_graphviz(clf, out_file=None, 
        #                         feature_names=x_train.columns,  
        #                         class_names=['ok','nok'])

        # # Visualize the decision tree using graphviz
        # graph = graphviz.Source(dot_data)
        # graph.render("decision_tree_graph", format="png")  # Save as PNG
        # mlflow.log_artifact("decision_tree_graph.png")

    return clf

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




