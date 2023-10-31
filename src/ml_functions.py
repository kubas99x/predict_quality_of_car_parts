import mlflow.keras
import mlflow
import mlflow.tensorflow

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

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