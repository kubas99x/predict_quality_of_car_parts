from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from mlflow import log_params, log_metrics, start_run
import mlflow
import mlflow.keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score
import os 

import numpy as np
import umap
from ml_models.neural_network_models import return_model
from ml_functions import *


def compile_fit_evaluate_model(x_train_, x_valid_, x_test_, y_train, y_valid, y_test, epochs_=10,
                               batch_size_=64, optimizer_='adam', comment='no comment', run_name_='standard_run',
                                model_number = 1, drop_neurons = 0.5, to_dict=False):
    
    x_train = np.copy(x_train_)
    x_valid = np.copy(x_valid_)
    x_test = np.copy(x_test_)

    # y_train = y_train['our_final_status'].squeeze().astype('float32')
    # y_test = y_test['our_final_status'].squeeze().astype('float32')
    # y_valid = y_valid['our_final_status'].squeeze().astype('float32')

    #metrics_ = ['accuracy', tf.keras.metrics.Recall(class_id=1, name='recall_1'),
    #            tf.keras.metrics.Recall(class_id=0, name='recall_0'), tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
    
    #metrics_ = ['accuracy', tf.keras.metrics.Recall(class_id=1, name='recall_1')]
    #metrics_ = ['accuracy', tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')]
    #metrics_ = ['accuracy']
    metrics_ = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(class_id=0, name='recall_0', thresholds=0.5),
                 tf.keras.metrics.SpecificityAtSensitivity(sensitivity= 0.5, name='specificity')]

    artifact_directory = "neural_network_v1"
    mlflow.set_experiment(artifact_directory)
    mlflow.tensorflow.autolog()

    with start_run(run_name=run_name_):
        
        model = return_model(model_number, x_train.shape[1], drop_=drop_neurons)

        # custom_optimizer = Adam(learning_rate=0.001)
        # Adam zajebiscie pasuje do dużych modeli a binary_crossentropy do binarnej klasyfikacji, 'adam'
        model.compile(loss='binary_crossentropy', optimizer=optimizer_, metrics=metrics_) 
        #callbackStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=5)

        #epoch - ilosc przejsc po calym datasecie, batch_size - ile wierszy jest branych w jednej iteracji
        #, callbacks = [callbackStopping]
        model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_, validation_data=(x_valid, y_valid)) 

        loss, accuracy, recall_0_, recall_1_ = model.evaluate(x_test, y_test)
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

# how to load model:
# loaded_model = mlflow.keras.load_model(r'C:\Users\dlxpmx8\Desktop\Projekt_AI\meb_process_data_analysis\src\mlruns\0\25857868653e497d806538cc98c80316\artifacts\model_test')

