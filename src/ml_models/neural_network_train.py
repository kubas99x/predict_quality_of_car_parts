import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def compile_fit_evaluate_model(model_, x_train, y_train, x_valid, y_valid, x_test, y_test, epochs_=10, batch_size_=64, optimizer_='adam', metrics_='accuracy'):
    # custom_optimizer = Adam(learning_rate=0.001)
    # Adam zajebiscie pasuje do dużych modeli a binary_crossentropy do binarnej klasyfikacji, 'adam'
    model_.compile(loss='binary_crossentropy', optimizer=optimizer_, metrics=[f'{metrics_}']) 

    #epoch - ilosc przejsc po calym datasecie, batch_size - ile wierszy jest branych w jednej iteracji
    model_.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_, validation_data=(x_valid, y_valid)) 

    loss, accuracy = model_.evaluate(x_test, y_test)

    return loss, accuracy