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
        'learning_rate': [0.1, 0.3],
        'max_depth': [8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'subsample': [0.9],
        'min_child_weight': range(2, 4),
        'gamma': [0],
        'random_state': [1011],
        'n_estimators': range(100, 600, 100),
        'booster': ['gbtree']
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
            model = XGBClassifier(**model_params)
            model.fit(x_train, y_train)
            mlflow.xgboost.log_model(model, "model")

            predictions = model.predict(x_valid)
            fig = distribution_of_probability_plot(predictions, y_valid)
            predictions = np.where(predictions > 0.5, 1, 0)
            recall_ok = recall_score(y_valid, predictions, pos_label=0)
            recall_nok = recall_score(y_valid, predictions, pos_label=1)
            accuracy = accuracy_score(y_valid, predictions)

            

            log_params(model_params)
            mlflow.log_figure(fig, 'model_probability.png')
            log_metrics({'recall_nok':recall_nok, 'recall_ok':recall_ok, 'acc_test':accuracy})
        mlflow.end_run()
        
    return model