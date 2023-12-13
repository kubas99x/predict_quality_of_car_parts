from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, models, regularizers


def return_model(model_number, shape, drop_ = 0.5):

    match model_number:
        case 0:
            model = Sequential()
            model.add(layers.Input(shape=(shape,)))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
        case 1:
            model = Sequential([
                    layers.Dense(64, activation='relu', input_shape=(shape,)),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(1, activation='sigmoid')   
                    ])
        case 2:
            model = Sequential([
                    layers.Dense(8, activation='relu', input_shape=(shape,)),
                    layers.Dense(1, activation='sigmoid')
                    ])
        case 3:
            model = Sequential([
                    layers.Dense(224, activation='relu', input_shape=(shape,)),
                    layers.Dense(224, activation='relu'),
                    layers.Dense(112, activation='relu'),
                    layers.Dense(66, activation='relu'),
                    layers.Dense(33, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                    ])
        case 4:
            model = Sequential([
                layers.Dense(224, activation='relu', input_shape=(shape,)),
                Dropout(drop_),
                layers.Dense(224, activation='relu'),
                Dropout(drop_),
                layers.Dense(112, activation='relu'),
                Dropout(drop_),
                layers.Dense(66, activation='relu'),
                layers.Dense(33, activation='relu'),
                layers.Dense(1, activation='sigmoid')
                ])
        case 5:
            model = Sequential()
            model.add(layers.Input(shape=(shape,)))
            model.add(layers.Dense(128, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
        case _:
            raise ValueError(f"Invalid model_number: {model_number}")
    
    return model

