import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from mlflow import log_params, log_metrics, start_run

def xgb_model(x_train, x_valid, x_test, y_train, y_valid, y_test, run_name_='standard_run', comment='no comment'):

    artifact_directory="xgboost"
    mlflow.set_experiment(artifact_directory)
    mlflow.xgboost.autolog()

    with start_run(run_name=run_name_):
        
        train = xgb.DMatrix(x_train, label=y_train)
        valid = xgb.DMatrix(x_valid, y_valid)
        test = xgb.DMatrix(x_test, y_test)

        cv = KFold(n_splits=5, shuffle=True, random_state=1011).split(X=x_train, y=y_train)

        grid_params = {
            'learning_rate': [0.05, 0.3],
            'max_depth': range(2, 9, 2),
            'colsample_bytree': [0.5, 1],
            'subsample': [0.9, 1],
            'min_child_weight': range(1, 5),
            'gamma': [0, 0.1],
            'random_state': [1011],
            'n_estimators': range(200, 2000, 200),
            'booster': ['gbtree'],
            'eval_metric': ['auc'],
            'objective': ['binary:logistic']
        }


        eval_parameters = {
            'early_stopping_rounds': 100,
            'eval_metric': 'auc',
            'eval_set': [(x_valid, y_valid)]
        }

        clf = XGBClassifier(objective='binary:logistic')

        grid_search = GridSearchCV(clf, param_grid=grid_params, n_jobs=3, scoring='accuracy', cv=cv)

        grid_search.fit(x_train, y_train, **eval_parameters)

        model = xgb.train(
            params = grid_search.best_params_,
            dtrain=train,
            num_boost_round = 600,
            evals=[(valid, 'eval'), (train, 'train')],
            verbose_eval=100
        )

        predictions = model.predict(test)
        predicted_classes = (predictions > 0.5).astype(int)  
        recall_nok = recall_score(y_test, predicted_classes, pos_label=1)
        recall_ok = recall_score(y_test, predicted_classes, pos_label=0)
        accuracy = accuracy_score(y_test, predicted_classes)

        log_params({'comment': comment, 'used_columns_shape':x_train.shape})
        log_metrics({'recall_nok':recall_nok, 'recall_ok':recall_ok, 'acc_test':accuracy})

    return grid_search.best_estimator_