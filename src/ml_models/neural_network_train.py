from mlflow import log_params, log_metrics, start_run
import mlflow
import mlflow.keras
from sklearn.metrics import accuracy_score, recall_score

import numpy as np
from ml_models.neural_network_models import return_model
from ml_functions import *


def compile_fit_evaluate_model(x_train_, x_valid_, x_test_, y_train, y_valid, y_test, epochs_=10,
                               batch_size_=64, optimizer_='adam',metrics_='accuracy', comment='no comment', run_name_='standard_run',
                                model_number = 1, drop_neurons = 0.5):
    
    x_train = np.copy(x_train_)
    x_valid = np.copy(x_valid_)
    x_test = np.copy(x_test_)
    
    artifact_directory = "neural_network"
    mlflow.set_experiment(artifact_directory)
    mlflow.tensorflow.autolog()

    with start_run(run_name=run_name_):
        
        model = return_model(model_number, x_test.shape[1], drop_=drop_neurons)

        model.compile(loss='binary_crossentropy', optimizer=optimizer_, metrics=[f'{metrics_}']) 
        model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_, validation_data=(x_valid, y_valid)) 

        loss, accuracy = model.evaluate(x_test, y_test)
        print(f'loss: {loss}')
        print(f'accuracy: {accuracy}')

        predictions = model.predict(x_test)
        predicted_classes = (predictions > 0.5).astype(int)  
        recall_class_1 = recall_score(y_test, predicted_classes, pos_label=1)
        recall_class_0 = recall_score(y_test, predicted_classes, pos_label=0)
        accuracy = accuracy_score(y_test, predicted_classes)
        
        # Log parameters and metrics to MLflow, no spaces allowed
        log_params({'comment': comment, 'used_columns_shape':x_train.shape})
        log_metrics({'recall_nok':recall_class_1, 'recall_ok':recall_class_0, 'acc_test':accuracy, 'loss_test':loss})

        fig = distribution_of_probability_plot(predictions, y_test)
        mlflow.log_figure(fig, 'model_probability.png')