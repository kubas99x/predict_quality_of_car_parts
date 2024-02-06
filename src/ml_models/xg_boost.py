import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, recall_score
from mlflow import log_params, log_metrics, start_run

from ml_functions import *

def xgb_model(x_train, x_valid, x_test, y_train, y_valid, y_test,  run_name_='standard_run', comment='no comment'):

    mlflow.set_experiment(run_name_)
    mlflow.xgboost.autolog()

    cv = KFold(n_splits=5, shuffle=True, random_state=1011).split(X=x_train, y=y_train)

    grid_params = {
        'learning_rate': [0.25, 0.275, 0.3],
        'max_depth': [5, 6, 7, 8],
        'colsample_bytree': [0.8, 0.9, 1],
        'subsample': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 2, 3],
        'gamma': [0, 0.1, 0.2],
        'random_state': [1011],
        'n_estimators': [50],
        'booster': ['gbtree'],
        'objective': ['binary:logistic']
    }

    evaluation_parameters = {'early_stopping_rounds': 20,
                            'eval_metric': 'auc',
                            'eval_set': [(x_valid, y_valid)]}

    clf = XGBClassifier(objective='binary:logistic')

    grid_search = GridSearchCV(clf, param_grid=grid_params, n_jobs=3, cv=cv)

    grid_search.fit(x_train, y_train, **evaluation_parameters)

    for i, model_params in enumerate(grid_search.cv_results_['params']):
        rn = f'xgboost_{i}'

        
        with mlflow.start_run(run_name=rn):
            # set and fit model
            model = XGBClassifier(**model_params)
            model.fit(x_train, y_train)
            mlflow.xgboost.log_model(model, "model")

            # predict propabilities
            predictions_proba = model.predict_proba(x_valid)
            predictions_proba = predictions_proba[:, 1]
            predictions_proba = predictions_proba.reshape(-1, 1)

            # create and save plot of distribution of propability
            fig = distribution_of_probability_plot(predictions_proba, y_valid)
            mlflow.log_figure(fig, 'model_probability.png')
            
            # predit classes
            predictions = model.predict(x_valid)
            predictions = np.where(predictions > 0.5, 1, 0)

            # create and save confusion matrix
            cm = create_confusion_matrix(y_valid, predictions)
            mlflow.log_figure(cm, 'confusion_matrix.png')

            # counting metrics
            recall_ok = recall_score(y_valid, predictions, pos_label=0)
            recall_nok = recall_score(y_valid, predictions, pos_label=1)
            accuracy = accuracy_score(y_valid, predictions)

            log_params(model_params)
            log_metrics({'recall_nok':recall_nok, 'recall_ok':recall_ok, 'acc_test':accuracy})
        mlflow.end_run()
    return model

def xgb_model_no_grid(x_train, x_valid, x_test, y_train, y_valid, y_test, model_name,  run_name_='standard_run', comment='no comment', threshold=0.9):

    mlflow.set_experiment(run_name_)
    mlflow.xgboost.autolog()

    # BEST PARAMS
    grid_params = {
        'learning_rate': 0.275,
        'max_depth': 7,
        'colsample_bytree': 0.9,
        'subsample': 0.8,
        'min_child_weight': 2,
        'gamma': 0,
        'random_state': 1011,
        'n_estimators': 50,
        'booster': 'gbtree',
        'objective': 'binary:logistic'
    }


    rn = f'xgboost_{model_name}_'

    with mlflow.start_run(run_name=rn):
        model = XGBClassifier(**grid_params)
        model.fit(x_train, y_train)
        mlflow.xgboost.log_model(model, "model")

        predictions_proba = model.predict_proba(x_valid)[:, 1].reshape(-1, 1)

        fig = distribution_of_probability_plot(predictions_proba, y_valid)
        mlflow.log_figure(fig, 'model_probability.png')

        predictions = np.where(predictions_proba > threshold, 1, 0)

        cm = create_confusion_matrix(y_valid, predictions)
        mlflow.log_figure(cm, 'confusion_matrix.png')

        recall_ok = recall_score(y_valid, predictions, pos_label=0)
        recall_nok = recall_score(y_valid, predictions, pos_label=1)
        accuracy = accuracy_score(y_valid, predictions)

        log_params(grid_params)
        log_metrics({'recall_nok': recall_nok, 'recall_ok': recall_ok, 'acc_test': accuracy})
    
    return model