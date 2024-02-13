import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from mlflow import log_params, log_metrics, start_run
import mlflow
import mlflow.keras

def random_forest_model(x_train, x_valid, x_test, y_train, y_valid, y_test, run_name_='standard_run', comment='no comment'):

    artifact_directory="random_forest"
    mlflow.set_experiment(artifact_directory)
    mlflow.sklearn.autolog()

    with start_run(run_name=run_name_):
        clf = RandomForestClassifier(random_state=0)

        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        predicted_classes = (predictions > 0.5).astype(int)  
        recall_nok = recall_score(y_test, predicted_classes, pos_label=1)
        recall_ok = recall_score(y_test, predicted_classes, pos_label=0)
        accuracy = accuracy_score(y_test, predictions)

        log_params({'comment': comment, 'used_columns_shape':x_train.shape})
        log_metrics({'recall_nok':recall_nok, 'recall_ok':recall_ok, 'acc_test':accuracy})

    return clf

