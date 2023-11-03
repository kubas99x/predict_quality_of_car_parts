from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers


def return_model(model_number, shape, drop_ = 0.5):

    match model_number:
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
            model = Sequential([
                    layers.Dense(64, activation='tanh', input_shape=(shape,)),
                    layers.Dense(64, activation='tanh'),
                    layers.Dense(1, activation='sigmoid')   
                    ])
        case 6:
            model = Sequential([
                    layers.Dense(224, activation='tanh', input_shape=(shape,)),
                    layers.Dense(224, activation='tanh'),
                    layers.Dense(112, activation='tanh'),
                    layers.Dense(66, activation='tanh'),
                    layers.Dense(33, activation='tanh'),
                    layers.Dense(1, activation='sigmoid')
                    ])
        case _:
            raise ValueError("Invalid model_number. Please use 1, 2, or 3.")
    
    return model

