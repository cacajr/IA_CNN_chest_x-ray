import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.cnnmodels import build_simple_cnn,build_tuned_cnn,build_model_vgg16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from skopt.space import Real, Categorical
from collections import Counter


def load_full_dataset(data_dir, target_size=(224, 224)):
    file_paths, labels = [], []
    classes = sorted(os.listdir(data_dir))
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(class_dir, file))
                labels.append(label)

    df = pd.DataFrame({'file_paths': file_paths, 'labels': labels})
    le = LabelEncoder()
    labels_encoded = le.fit_transform(df['labels'])

    X = np.array([
        img_to_array(load_img(path, target_size=target_size)) / 255.0
        for path in df['file_paths']
    ])
    y = np.array(labels_encoded)
    return X, y, le, df

def limpar_memoria_keras(objetos_a_deletar=None, verbose=True):
    """
    Libera memória da GPU e da RAM após treinos com Keras.

    Parâmetros:
    - objetos_a_deletar: lista de variáveis a deletar (ex: [grid, model])
    - verbose: se True, imprime confirmação

    Exemplo de uso:
    limpar_memoria_keras([grid, model])
    """
    if objetos_a_deletar:
        for obj in objetos_a_deletar:
            del obj

    K.clear_session()
    gc.collect()
    
    if verbose:
        print("Memória do Keras liberada com sucesso.")

def split_dataset(X, y, seed, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

def create_data_generators(X_train, y_train, X_val, y_val, batch_size):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator, epochs, callbacks):
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test,y):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_proba = model.predict(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class='ovr')


    return acc, f1, auc, 

def treinar_modelo_multiseed(X, y, model_builder, best_params, callbacks, seeds, epochs=10):
    metrics_list = []
    histories = []
    models = []

    for seed in seeds:
        print(f"\nTreinando com seed {seed}")
        tf.random.set_seed(seed)
        np.random.seed(seed)

        X_train, X_test, y_train, y_test = split_dataset(X, y, seed)
        train_generator, val_generator = create_data_generators(
            X_train, y_train, X_test, y_test, best_params["batch_size"]
        )

        model = model_builder(
            dropout_rate=best_params["model__dropout_rate"],
            learning_rate=best_params["model__learning_rate"]
        )

        history = train_model(model, train_generator, val_generator, epochs, callbacks)
        acc, f1, auc, report, cf_matrix  = evaluate_model(model, X_test, y_test,y)

        print(f"Seed {seed} | ACC: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        metrics_list.append((acc, f1, auc, report, cf_matrix))
        histories.append(history)
        models.append(model)


    return metrics_list, histories, models

def encontrar_modelo_representativo(models, metrics_list):
    """
    Retorna o índice e o modelo com métricas mais próximas da média geral.

    Parâmetros:
    - models: lista de modelos treinados
    - metrics_list: lista de tuplas (acc, f1, auc) para cada modelo

    Retorno:
    - idx: índice do modelo representativo
    - modelo: o modelo representativo
    """
    metrics_array = np.array(metrics_list)
    mean_metrics = np.mean(metrics_array, axis=0)

    # Distância euclidiana entre as métricas de cada modelo e a média
    distances = np.linalg.norm(metrics_array - mean_metrics, axis=1)
    idx = np.argmin(distances)

    print(f"Modelo representativo: índice {idx}")
    print(f"Métricas do modelo: ACC={metrics_list[idx][0]:.4f}, F1={metrics_list[idx][1]:.4f}, AUC={metrics_list[idx][2]:.4f}")
    print(f"Métricas médias:     ACC={mean_metrics[0]:.4f}, F1={mean_metrics[1]:.4f}, AUC={mean_metrics[2]:.4f}")
    
    return idx, models[idx], metrics_list[idx]