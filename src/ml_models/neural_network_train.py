import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("nueral_network_experiment")
ex.observers.append(FileStorageObserver(r'/neural_models'))

@ex.config
def my_config():
    epochs_=10
    batch_size_=64
    optimizer_='adam'
    metrics_='accuracy'

@ex.capture
def prepare_message(message):
    return message

@ex.main
def compile_fit_evaluate_model(x_train, x_valid, x_test, y_train, y_valid, y_test, epochs_=10):
    print(x_train.shape[1])
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

    model.save(r'./model_1.h5')
    ex.add_artifact(r'model_1.h5')

ex.add_source_file(r"C:\Users\dlxpmx8\Desktop\Projekt_AI\meb_process_data_analysis\src\ml_models\neural_network_train.py")

#ex.run()