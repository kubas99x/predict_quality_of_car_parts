import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from mlflow import log_params, log_metrics, log_artifact, start_run

def compile_fit_evaluate_model(x_train, x_valid, x_test, y_train, y_valid, y_test, epochs_=10,
                               batch_size_=64, optimizer_='adam',metrics_='accuracy'):
    
    with start_run():
        model = Sequential([
        layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')   # Probability score between 0 and 1
        ])
        
        # custom_optimizer = Adam(learning_rate=0.001)
        # Adam zajebiscie pasuje do dużych modeli a binary_crossentropy do binarnej klasyfikacji, 'adam'
        model.compile(loss='binary_crossentropy', optimizer=optimizer_, metrics=[f'{metrics_}']) 

        #epoch - ilosc przejsc po calym datasecie, batch_size - ile wierszy jest branych w jednej iteracji
        model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_, validation_data=(x_valid, y_valid)) 

        loss, accuracy = model.evaluate(x_test, y_test)
        print(f'loss: {loss}')
        print(f'accuracy: {accuracy}')

        # Save the model
        model.save('neural_models/model_1')
        
        # Log parameters and metrics to MLflow
        log_params({'epochs': epochs_, 'batch_size': batch_size_, 'optimizer': optimizer_, 'metrics': metrics_})
        log_metrics({'loss': loss, 'accuracy': accuracy})
        
        # Log the saved model as an artifact
        log_artifact('neural_models/model_1', artifact_path='ml_models')


