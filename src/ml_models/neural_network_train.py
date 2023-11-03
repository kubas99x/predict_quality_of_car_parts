from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from mlflow import log_params, log_metrics, start_run
import mlflow
import mlflow.keras
from sklearn.metrics import accuracy_score, recall_score
import os 
from sklearn.decomposition import PCA
import numpy as np
import umap


def compile_fit_evaluate_model(x_train_, x_valid_, x_test_, y_train, y_valid, y_test, epochs_=10,
                               batch_size_=64, optimizer_='adam',metrics_='accuracy', comment='no comment', run_name_='standard_run',
                               n_components = None, n_components_umap = None, umap_min_dist = 0.1 ):
    
    # umap_min_dist=0.1
    
    x_train = np.copy(x_train_)
    x_valid = np.copy(x_valid_)
    x_test = np.copy(x_test_)
    
    artifact_directory = "neural_network"
    mlflow.set_experiment(artifact_directory)
    mlflow.tensorflow.autolog()

    if n_components is not None:
        pca = PCA(n_components=n_components)
        x_train = pca.fit_transform(x_train)
        x_valid = pca.transform(x_valid)
        x_test = pca.transform(x_test)
    
    if n_components_umap is not None:
        umap_model = umap.UMAP(n_components=n_components_umap, min_dist=umap_min_dist)
        x_train = umap_model.fit_transform(x_train)
        x_valid = umap_model.transform(x_valid)
        x_test = umap_model.transform(x_test)

    with start_run(run_name=run_name_):
        model = Sequential([
        layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')   # Probability score between 0 and 1
        ])

        # model = Sequential([
        # layers.Dense(8, activation='relu', input_shape=(x_train.shape[1],)),
        # layers.Dense(1, activation='sigmoid')
        # ])

        # custom_optimizer = Adam(learning_rate=0.001)
        # Adam zajebiscie pasuje do dużych modeli a binary_crossentropy do binarnej klasyfikacji, 'adam'
        model.compile(loss='binary_crossentropy', optimizer=optimizer_, metrics=[f'{metrics_}']) 

        #epoch - ilosc przejsc po calym datasecie, batch_size - ile wierszy jest branych w jednej iteracji
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

        #mlflow.keras.log_model(model, "model_saved")


# how to load model:
# loaded_model = mlflow.keras.load_model(r'C:\Users\dlxpmx8\Desktop\Projekt_AI\meb_process_data_analysis\src\mlruns\0\25857868653e497d806538cc98c80316\artifacts\model_test')

