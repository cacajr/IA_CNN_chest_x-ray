import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score, roc_curve, auc

import gc


# -----------------------
# Funções de pré-processamento
# -----------------------

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


def split_dataset(X, y, seed, test_size=0.2):
    """
    Divide o dataset em treino e teste de forma estratificada.

    Parâmetros:
    - X: array de imagens
    - y: array de rótulos
    - seed: valor para random_state
    - test_size: proporção do conjunto de teste

    Retorno:
    - X_train, X_test, y_train, y_test: dados divididos
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def add_salt_pepper_noise(image, amount=0.01, salt_vs_pepper=0.5):
    """
    Aplica ruído do tipo 'sal e pimenta' em uma imagem.

    Parâmetros:
    - image: array da imagem
    - amount: proporção de pixels afetados
    - salt_vs_pepper: proporção de pixels brancos (sal) em relação aos pretos (pimenta)

    Retorno:
    - imagem com ruído aplicado
    """
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy


def create_data_generators(X_train, y_train, X_test, y_test, batch_size):
    """
    Cria geradores de dados com data augmentation para treino e validação.

    Parâmetros:
    - X_train, y_train: dados de treino
    - X_test, y_test: dados de validação
    - batch_size: tamanho do lote

    Retorno:
    - train_generator, val_generator: geradores de dados
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        preprocessing_function=lambda img: add_salt_pepper_noise(img, amount=0.01)
    )
    datagen.fit(X_train)

    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=batch_size)
    return train_generator, val_generator


def create_data_generators_robustness(X_test, y_test, batch_size):
    """
    Cria um gerador de dados com perturbações para teste de robustez.

    Parâmetros:
    - X_test, y_test: dados de teste
    - batch_size: tamanho do lote

    Retorno:
    - test_generator: gerador com ruído para avaliação
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        preprocessing_function=lambda img: add_salt_pepper_noise(img, amount=0.01)
    )
    datagen.fit(X_test)
    test_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
    return test_generator



# -----------------------
# Treinamento e avaliação
# -----------------------

def limpar_memoria_keras(objetos=[], verbose=True):
    """
    Libera memória ocupada por objetos do Keras e limpa a sessão atual.

    Parâmetros:
    - objetos: lista de objetos a serem deletados (modelos, históricos etc.)
    - verbose: se True, imprime mensagem
    """
    if verbose:
        print("Limpando memória do Keras...")
    for obj in objetos:
        del obj
    gc.collect()
    K.clear_session()


def evaluate_model(model, X_test, y_test, y):
    """
    Avalia um modelo treinado em dados de teste.

    Parâmetros:
    - model: modelo treinado
    - X_test, y_test: dados de teste
    - y: array com todas as classes possíveis (usado para binarização)

    Retorno:
    - acc: acurácia
    - f1: F1-score macro
    - auc_score: AUC macro multiclasse
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_proba = model.predict(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auc_score = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class='ovr')
    return acc, f1, auc_score


def treinar_modelo_multiseed(X, y, model_builder, best_params, seeds, epochs=10, model_name="model"):
    """
    Treina múltiplos modelos com diferentes seeds para avaliar robustez.

    Parâmetros:
    - X, y: dados de entrada
    - model_builder: função que constrói o modelo
    - best_params: melhores hiperparâmetros encontrados
    - seeds: lista de sementes aleatórias
    - epochs: número de épocas
    - model_name: prefixo para salvar arquivos

    Retorno:
    - metrics_list: lista com métricas [acc, f1, auc]
    - histories: lista com históricos de treinamento
    - model_paths: caminhos dos modelos salvos
    """
    metrics_list, histories, model_paths = [], [], []

    for seed in seeds:
        print(f"\nTreinando com seed {seed}")
        tf.random.set_seed(seed)
        np.random.seed(seed)

        X_train, X_test, y_train, y_test = split_dataset(X, y, seed)
        train_gen, val_gen = create_data_generators(X_train, y_train, X_test, y_test, best_params["batch_size"])

        model = model_builder(
            dropout_rate=best_params["dropout_rate"],
            learning_rate=best_params["learning_rate"]
        )

        callbacks = [
            EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(f"{model_name}_seed_{seed}.keras", save_best_only=True, monitor='val_loss', verbose=0)
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        with open(f'{model_name}_seed_{seed}.json', 'w') as f:
            json.dump(history.history, f)

        acc, f1, auc_score = evaluate_model(model, X_test, y_test, y)
        print(f"Seed {seed} | ACC: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")

        metrics_list.append([acc, f1, auc_score])
        histories.append(history)
        model_paths.append(f"{model_name}_seed_{seed}.keras")

        limpar_memoria_keras([model, train_gen, val_gen, history], verbose=False)

    return metrics_list, histories, model_paths


def encontrar_modelo_representativo(models, metrics_list):
    """
    Retorna o modelo cuja performance mais se aproxima da média das métricas.

    Parâmetros:
    - models: lista de caminhos dos modelos salvos
    - metrics_list: lista de métricas [acc, f1, auc] por modelo

    Retorno:
    - idx: índice do modelo mais representativo
    - modelo_path: caminho do modelo representativo
    - metrics: métricas do modelo representativo
    """
    metrics_array = np.array(metrics_list)
    mean_metrics = np.mean(metrics_array, axis=0)

    distances = np.linalg.norm(metrics_array - mean_metrics, axis=1)
    idx = np.argmin(distances)

    print(f"Modelo representativo: índice {idx}")
    print(f"Métricas do modelo: ACC={metrics_list[idx][0]:.4f}, F1={metrics_list[idx][1]:.4f}, AUC={metrics_list[idx][2]:.4f}")
    print(f"Métricas médias:     ACC={mean_metrics[0]:.4f}, F1={mean_metrics[1]:.4f}, AUC={mean_metrics[2]:.4f}")

    return idx, models[idx], metrics_list[idx]


# -----------------------
# Visualizações
# -----------------------

def plot_sample_images_per_class(X, y, label_encoder, samples_per_class=5, figsize=(15, 10)):
    """
    Plota algumas imagens de exemplo para cada classe.

    Parâmetros:
    - X: array de imagens
    - y: array de rótulos
    - label_encoder: encoder para converter rótulos para nomes
    - samples_per_class: número de amostras por classe a serem exibidas
    - figsize: tamanho da figura
    """
    classes = np.unique(y)
    fig, axs = plt.subplots(len(classes), samples_per_class, figsize=figsize)

    for i, label in enumerate(classes):
        label_name = label_encoder.inverse_transform([label])[0]
        idxs = np.where(y == label)[0][:samples_per_class]
        for j, idx in enumerate(idxs):
            axs[i, j].imshow(X[idx])
            axs[i, j].axis('off')
            axs[i, j].set_title(label_name if j == 0 else "")
    
    plt.tight_layout()
    plt.show()

def plot_multiclass_roc(y_true, y_score, classes, figsize=(10, 8)):
    """
    Plota as curvas ROC para classificação multiclasse (One-vs-Rest).

    Parâmetros:
    - y_true: rótulos verdadeiros
    - y_score: probabilidades previstas (modelo.predict_proba)
    - classes: array de classes possíveis
    - figsize: tamanho do gráfico
    """

    y_test_bin = label_binarize(y_true, classes=classes)
    n_classes = y_test_bin.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=figsize)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Classe {classes[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot(fpr["macro"], tpr["macro"], label=f"Média macro (AUC = {roc_auc['macro']:.2f})", color='black', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curvas ROC - Classificação Multiclasse (OvR)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_history(history, title_prefix="Modelo"):
    """
    Plota a acurácia e o erro (loss) durante o treinamento.

    Parâmetros:
    - history: objeto retornado por model.fit()
    - title_prefix: prefixo do título do gráfico
    """
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Acurácia Treino')
    plt.plot(epochs, val_acc, 'ro-', label='Acurácia Validação')
    plt.title(f'{title_prefix} - Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Loss Treino')
    plt.plot(epochs, val_loss, 'ro-', label='Loss Validação')
    plt.title(f'{title_prefix} - Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def kfold_search_hyperparameters(X, y, model_func, param_grid, k=4, random_state=7, output_json=True):
    """
    Realiza busca de hiperparâmetros usando validação cruzada com KFold.

    Parâmetros:
    - X, y: dados de entrada e rótulos
    - model_func: função que retorna um modelo compilado (ex: build_model_vgg16)
    - param_grid: dicionário com listas de valores para dropout_rate, learning_rate e batch_size
    - k: número de folds
    - random_state: para reprodutibilidade
    - output_json: se True, salva melhores parâmetros em arquivo JSON

    Retorno:
    - best_params: dicionário com melhores hiperparâmetros encontrados
    - best_score: F1 médio correspondente
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    parameters_callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
    ]

    best_score = -1
    best_params = None

    for dropout in param_grid['dropout_rate']:
        for lr in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                print(f"Testing: dropout={dropout}, lr={lr}, batch_size={batch_size}")
                f1_scores = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    print(f"Fold {fold+1}/{k}")
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model = model_func(dropout_rate=dropout, learning_rate=lr)

                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=parameters_callbacks,
                        verbose=0
                    )

                    y_pred = model.predict(X_val).argmax(axis=1)
                    y_true = y_val if y_val.ndim == 1 else y_val.argmax(axis=1)

                    f1 = f1_score(y_true, y_pred, average='macro')
                    f1_scores.append(f1)

                    print(f"F1 (fold {fold+1}): {f1:.4f}")
                    limpar_memoria_keras([model, history], verbose=False)

                avg_f1 = np.mean(f1_scores)
                print(f" Média F1: {avg_f1:.4f}")

                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_params = {
                        'dropout_rate': dropout,
                        'learning_rate': lr,
                        'batch_size': batch_size
                    }

                    if output_json:
                        filename = f"{model_func.__name__}_best_params.json"
                        with open(filename, 'w') as f:
                            json.dump(best_params, f)
                        print(f"Novo melhor conjunto salvo em {filename}")

    print(f"\nMelhores hiperparâmetros para {model_func.__name__}:")
    print(best_params)
    print(f"Com F1 médio: {best_score:.4f}")

    return best_params, best_score