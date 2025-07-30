import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
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

def split_validation(X_train, y_train, seed, val_size=0.2):
    """
    Divide o conjunto de treino em treino e validação de forma estratificada.

    Parâmetros:
    - X_train, y_train: dados já divididos com `split_dataset`
    - val_size: proporção para validação 
    - seed: random_state para reprodutibilidade

    Retorno:
    - X_train_final, X_val, y_train_final, y_val
    """

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=seed
    )
    return X_train_final, X_val, y_train_final, y_val


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


def create_data_generators(X_train, y_train, X_val, y_val, batch_size):
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
    val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size)
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
    # Shuffle False para manter a ordem.
    test_generator = datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)
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

        #Treinamento 64%
        #Validação	16%
        #Teste	    20%

        X_train, X_test, y_train, y_test = split_dataset(X, y, seed) # 80% treino, 20% teste
        X_train, X_val, y_train, y_val = split_validation(X_train, y_train, seed) # 20% de X_train para val
        train_gen, val_gen = create_data_generators(X_train,y_train, X_val, y_val, best_params["batch_size"])

        model = model_builder(
            dropout_rate=best_params["dropout_rate"],
            learning_rate=best_params["learning_rate"]
        )

        callbacks = [
            EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(f"../models/{model_name}_seed_{seed}.keras", save_best_only=True, monitor='val_loss', verbose=0)
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        with open(f'./history/{model_name}_seed_{seed}.json', 'w') as f:
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

def avaliar_robustez(X, y, model, nome, seed, batch_size=32):
    """
    Avalia a robustez do modelo com dados perturbados.
    
    Parâmetros:
    - model: modelo Keras treinado
    - nome: nome do modelo (para print)
    - seed: seed usada no split do modelo representativo
    - batch_size: tamanho do batch (default = 32)
    """
    print(f"\n==== Teste de Robustez: {nome} (seed={seed}) ====")
    
    # Usar mesmo conjunto de teste usado na avaliação do modelo representativo
    _, X_test, _, y_test = split_dataset(X, y, seed=seed)
    
    # Criar gerador com perturbações
    robust_gen = create_data_generators_robustness(X_test, y_test, batch_size=batch_size)

    # Avaliação
    loss, acc = model.evaluate(robust_gen, verbose=0)
    y_pred_robust = model.predict(robust_gen).argmax(axis=1)
    f1 = f1_score(y_test, y_pred_robust, average='macro')
    print(classification_report(y_test,y_pred_robust))

    print(f"Robust Accuracy: {acc:.4f}")
    print(f"Robust F1-score: {f1:.4f}")


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
    """
    acc = history.get('accuracy')
    val_acc = history.get('val_accuracy')
    loss = history.get('loss')
    val_loss = history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Acurácia Treino', markersize=2)
    plt.plot(epochs, val_acc, 'ro-', label='Acurácia Validação', markersize=2)
    plt.title(f'{title_prefix} - Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Loss Treino', markersize=2)
    plt.plot(epochs, val_loss, 'ro-', label='Loss Validação', markersize=2)
    plt.title(f'{title_prefix} - Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix_heatmap(y_true, y_pred, model_name, class_names=None):
    """
    Plota uma matriz de confusão como heatmap.

    Parâmetros:
    - y_true: rótulos verdadeiros
    - y_pred: rótulos previstos
    - class_names: lista com nomes das classes (opcional)
    - model_name: nome do modelo para título
    """
    title = f"Matriz de Confusão - {model_name}"
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto"
    )

    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------
# Encontrar Hiperparâmetros
# -----------------------

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

    X_t, _, y_t, _ = split_dataset(X, y, random_state)

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
                    X_train, X_val = X_t[train_idx], X_t[val_idx]
                    y_train, y_val = y_t[train_idx], y_t[val_idx]

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
                        filename = f"./params/{model_func.__name__}_best_params.json"
                        with open(filename, 'w') as f:
                            json.dump(best_params, f)
                        print(f"Novo melhor conjunto salvo em {filename}")

    print(f"\nMelhores hiperparâmetros para {model_func.__name__}:")
    print(best_params)
    print(f"Com F1 médio: {best_score:.4f}")

    return best_params, best_score

def avaliar_modelo_base(model, seed, X, y):
    X_train, X_test, y_train, y_test = split_dataset(X, y, seed=seed)
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_proba = model.predict(X_test)
    return y_test, y_pred, y_proba, y_test_bin

# -----------------------
# Carregamento de modelos e métricas
# -----------------------

def salvar_resumo_metricas_modelos(
    simple_metrics, simple_model_metrics,
    tuned_metrics, tuned_model_metrics,
    vgg_metrics, vgg16_model_metrics,
    simple_idx, tuned_idx, vgg16_idx,
    output_path="resumo_metricas_modelos.json"
    ):
    """
    Gera e salva um resumo com as métricas médias e representativas dos modelos treinados.

    Parâmetros:
    - *_metrics: listas de métricas por seed (ex: [[acc, f1, auc], ...])
    - *_model_metrics: métricas do modelo representativo (ex: [acc, f1, auc])
    - *_idx: índice da seed escolhida como representativa
    - output_path: nome do arquivo JSON de saída

    Retorno:
    - dicionário com os dados salvos
    """
    metricas = {
        'mean_simple_metrics': np.mean(np.asarray(simple_metrics), axis=0).tolist(),
        'model_simple_metrics': simple_model_metrics,
        'mean_tuned_metrics': np.mean(np.asarray(tuned_metrics), axis=0).tolist(),
        'model_tuned_metrics': tuned_model_metrics,
        'mean_vgg16_metrics': np.mean(np.asarray(vgg_metrics), axis=0).tolist(),
        'model_vgg16_metrics': vgg16_model_metrics,
        'chosen_seeds': [simple_idx, tuned_idx, vgg16_idx]
    }
    with open(output_path, "w") as f:
        json.dump(metricas, f, indent=4)

def carregar_modelos_representativos(json_path="resumo_metricas_modelos.json"):
    """
    Carrega os modelos representativos e seus históricos (history) para simple, tuned e vgg16 com base no JSON de resumo.

    Retorno:
    - models: dicionário com os três modelos carregados
    - histories: dicionário com os history dos modelos
    - seeds: dicionário com os índices representativos usados
    - paths: dicionário com os caminhos dos arquivos carregados
    """
    with open(json_path, "r") as f:
        resumo = json.load(f)

    # Ajustar índices (caso estejam baseados em 0 no JSON)
    simple_idx = resumo['chosen_seeds'][0] +1
    tuned_idx = resumo['chosen_seeds'][1] +1
    vgg16_idx = resumo['chosen_seeds'][2] +1

    # Caminhos para modelos
    paths = {
        "simple": f"../models/simple_seed_{simple_idx}.keras",
        "tuned": f"../models/tuned_seed_{tuned_idx}.keras",
        "vgg16": f"../models/vgg16_seed_{vgg16_idx}.keras"
    }

    # Caminhos para históricos
    history_paths = {
        "simple": f"./history/simple_seed_{simple_idx}.json",
        "tuned": f"./history/tuned_seed_{tuned_idx}.json",
        "vgg16": f"./history/vgg16_seed_{vgg16_idx}.json"
    }

    # Índices usados
    seeds = {
        "simple": simple_idx,
        "tuned": tuned_idx,
        "vgg16": vgg16_idx
    }

    # Carregar modelos
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = load_model(path)
            print(f"Modelo '{name}' carregado de {path}")
        else:
            models[name] = None
            print(f"Caminho do modelo não encontrado: {path}")

    # Carregar históricos
    histories = {}
    for name, hist_path in history_paths.items():
        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                histories[name] = json.load(f)
            print(f"Histórico '{name}' carregado de {hist_path}")
        else:
            histories[name] = None
            print(f"Histórico não encontrado: {hist_path}")

    return models, histories, seeds, paths


####################
##### GRAD-CAM #####
####################
def get_layer_by_name(model, name):
    """
    Suporte a caminhos hierárquicos.
    """
    if '/' in name:
        parts = name.split('/')
        submodel = get_layer_by_name(model, parts[0])
        return get_layer_by_name(submodel, '/'.join(parts[1:]))

    for layer in model.layers:
        if layer.name == name:
            return layer
        if isinstance(layer, tf.keras.Model):
            try:
                return get_layer_by_name(layer, name)
            except ValueError:
                continue
    raise ValueError(f"Nenhuma camada com nome '{name}' foi encontrada.")

def generate_gradcam(model, img_array, class_index, layer_name):
    """
    Gera o mapa de ativação Grad-CAM para uma imagem e classe específica.
    """
    # Captura a camada desejada (suporte a modelos aninhados como VGG16)
    if layer_name is None:
        # Pegamos a última camada Conv2D automaticamente
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("Nenhuma camada Conv2D encontrada no modelo.")
        layer_name = conv_layers[-1].name

    target_layer = get_layer_by_name(model, layer_name)

    # Força a execução para garantir que os tensores estejam definidos
    _ = model(img_array)

    grad_model = tf.keras.models.Model(
        [model.inputs], [target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4, cmap='jet'):
    """
    Sobrepõe o heatmap Grad-CAM à imagem original.

    Parâmetros:
    - img: imagem original (H, W, C), normalizada
    - heatmap: mapa Grad-CAM (H, W)
    - alpha: transparência da sobreposição

    Retorno:
    - imagem resultante (np.array)
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.uint8(img * 255), 1 - alpha, colormap, alpha, 0)
    return superimposed / 255.0

def plot_gradcam_for_sample(model, X, y_true, idx, label_encoder, title=None, layer_name=None):
    """
    Plota a imagem original, o heatmap bruto e o Grad-CAM sobreposto para uma amostra específica.

    Parâmetros:
    - model: modelo Keras
    - X: conjunto de imagens normalizadas
    - y_true: rótulos reais (inteiros)
    - idx: índice da amostra
    - label_encoder: para converter índice em string
    - title: título opcional
    - layer_name: nome da camada Conv2D alvo
    """
    img_input = np.expand_dims(X[idx], axis=0)
    img = X[idx]

    pred = model.predict(img_input, verbose=0)
    pred_class = np.argmax(pred)
    true_class = y_true[idx]

    print(f'Pred: {pred_class}, True: {true_class}')

    heatmap = generate_gradcam(model, img_input, class_index=pred_class, layer_name=layer_name)
    overlay = overlay_gradcam(img, heatmap)

    # Plota os 3: imagem original, raw heatmap e Grad-CAM
    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Imagem original")

    # Grad-CAM
    plt.subplot(1, 3, 2)
    plt.imshow(overlay)
    plt.axis('off')
    if title is None:
        title = f"Grad-CAM | Verdadeiro: {label_encoder.classes_[true_class]} | Previsto: {label_encoder.classes_[pred_class]}"
    plt.title(title)

    # Raw heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title("Raw heatmap")

    plt.tight_layout()
    plt.show()

def plot_gradcam_for_vgg16(
    gradcam_model,            # submodelo com as camadas convolucionais (ex: functional_0)
    full_model,               # modelo completo (com a cabeça de classificação)
    X,
    y_true,
    idx,
    label_encoder,
    title=None,
    layer_name="block5_conv3"
):
    img_input = np.expand_dims(X[idx], axis=0)
    img = X[idx]

    pred = full_model.predict(img_input, verbose=0)
    pred_class = np.argmax(pred)
    true_class = y_true[idx]

    print(f'Pred: {pred_class}, True: {true_class}')

    heatmap = generate_gradcam(gradcam_model, img_input, class_index=pred_class, layer_name=layer_name)
    overlay = overlay_gradcam(img, heatmap)

    # Plota os 3: imagem original, raw heatmap e Grad-CAM
    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Imagem original")

    # Grad-CAM
    plt.subplot(1, 3, 2)
    plt.imshow(overlay)
    plt.axis('off')
    if title is None:
        title = f"Grad-CAM | Verdadeiro: {label_encoder.classes_[true_class]} | Previsto: {label_encoder.classes_[pred_class]}"
    plt.title(title)

    # Raw heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title("Raw heatmap")

    plt.tight_layout()
    plt.show()

def convert_sequential_to_functional(sequential_model):
    """
    Converts a Keras Sequential model to a Functional model,
    ensuring unique layer names and proper weight transfer.
    """
    input_tensor = tf.keras.Input(shape=sequential_model.input_shape[1:], name="unique_input")
    x = input_tensor

    for i, layer in enumerate(sequential_model.layers):
        config = layer.get_config()
        config["name"] = f"{layer.__class__.__name__.lower()}_{i}"  # nome único
        cloned_layer = layer.__class__.from_config(config)
        x = cloned_layer(x)  # constroi a camada (build)
        cloned_layer.set_weights(layer.get_weights())  

    return tf.keras.Model(inputs=input_tensor, outputs=x)