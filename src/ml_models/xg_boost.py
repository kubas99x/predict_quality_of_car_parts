import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from mlflow import log_params, log_metrics, start_run

from ml_functions import *

def xgb_model(x_train, x_valid, x_test, y_train, y_valid, y_test, run_name_='standard_run', comment='no comment'):

    mlflow.set_experiment('xgboost_lwd')
    mlflow.xgboost.autolog()

    cv = KFold(n_splits=5, shuffle=True, random_state=1011).split(X=x_train, y=y_train)

    grid_params = {
        'learning_rate': np.linspace(0.25, 0.3, 5),
        'max_depth': [7],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'subsample': [0.8, 0.9],
        'min_child_weight': [2,3],
        'gamma': [0],
        'random_state': [1011],
        'n_estimators': [200],
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
        rn = f'xgboost_model_{i}_nn'

        
        with mlflow.start_run(run_name=rn):
            # set and fit model
            model = XGBClassifier(**model_params)
            model.fit(x_train, y_train)
            mlflow.xgboost.log_model(model, "model")

            # predict propabilities
            predictions_proba = model.predict_proba(x_valid)
            predictions_proba = predictions_proba[:, 1]
            predictions_proba = predictions_proba.reshape(-1, 1)

            # create and save 
            fig = distribution_of_probability_plot(predictions_proba, y_valid)
            mlflow.log_figure(fig, 'model_probability.png')
            
            # predit classes
            predictions = model.predict(x_valid)
            predictions = np.where(predictions > 0.5, 1, 0)

            # counting metrics
            recall_ok = recall_score(y_valid, predictions, pos_label=0)
            recall_nok = recall_score(y_valid, predictions, pos_label=1)
            accuracy = accuracy_score(y_valid, predictions)

            log_params(model_params)
            log_metrics({'recall_nok':recall_nok, 'recall_ok':recall_ok, 'acc_test':accuracy})
        mlflow.end_run()
        
    return model