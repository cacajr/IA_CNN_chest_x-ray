import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def load_data(data_dir, input_shape=(150, 150), batch_size=32, seed=42):
    """
    Carrega os dados de imagem e retorna os geradores de treino, validação e teste.

    Parâmetros:
    - data_dir (str): Caminho para o diretório com os subdiretórios train/val/test.
    - input_shape (tuple): Tamanho das imagens.
    - batch_size (int): Tamanho do batch.
    - seed (int): Semente para reprodutibilidade.

    Retorno:
    - train_generator, val_generator, test_generator
    """
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        directory=f"{data_dir}/train",
        target_size=input_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=seed
    )

    val_generator = datagen.flow_from_directory(
        directory=f"{data_dir}/val",
        target_size=input_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = datagen.flow_from_directory(
        directory=f"{data_dir}/test",
        target_size=input_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def train_model(model, train_generator, val_generator, epochs=10):
    """
    Treina um modelo Keras com os dados fornecidos.

    Parâmetros:
    - model: Instância do modelo Keras já compilado.
    - train_generator: Gerador com os dados de treino.
    - val_generator: Gerador com os dados de validação.
    - epochs (int): Número de épocas para treinamento.

    Retorno:
    - history: Objeto History do Keras contendo métricas de desempenho ao longo das épocas.
    """
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=1
    )

    return history

def evaluate_model(model, generator, class_names=None, threshold=0.5, plot_roc=True):
    """
    Avalia um modelo treinado com métricas de classificação e curva ROC.

    Parâmetros:
    - model (tf.keras.Model): Modelo treinado.
    - generator (DirectoryIterator ou semelhante): Gerador de dados com labels reais.
    - class_names (list): Nomes das classes, ex: ['NORMAL', 'PNEUMONIA']. Se None, usa índices.
    - threshold (float): Limite de decisão para converter probabilidades em classes.
    - plot_roc (bool): Se True, exibe o gráfico da curva ROC.

    Retorna:
    - dict: Dicionário contendo y_true, y_pred, y_pred_classes, auc e relatório.
    """
    # Previsões
    y_pred = model.predict(generator)
    y_pred_classes = (y_pred > threshold).astype(int).flatten()
    y_true = generator.classes

    # Nomes de classes
    if class_names is None:
        class_names = [str(i) for i in sorted(set(y_true))]

    # Métricas
    print("\nRelatório de Classificação:")
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)

    auc = roc_auc_score(y_true, y_pred)
    print(f"\nAUC-ROC: {auc:.4f}")

    # Curva ROC
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
        plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "y_true": y_true,
        "y_pred": y_pred.flatten(),
        "y_pred_classes": y_pred_classes,
        "auc": auc,
        "confusion_matrix": cm,
        "classification_report": report
    }

def plot_learning_curves(
        accuracies=[], 
        val_accuracies=[], 
        losses=[], 
        val_losses=[],
        model_name="Model"):
    """
    Plota os gráficos de perda (loss) e acurácia (accuracy) para treino e validação.

    Parâmetros:
    - accuracies (list): Acurácias dos treinos.
    - val_accuracies (list): Acurácias das validações.
    - losses (list): Perdas dos treinos.
    - val_losses (list): Perdas das validações.
    - model_name (str): Nome do modelo (usado nos títulos dos gráficos).
    """
    epochs = range(1, len(accuracies) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracies, 'bo-', label='Training acc')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def summarize_metrics(accuracies=[], losses=[], model_name="Modelo"):
    """
    Exibe estatísticas resumidas (média e desvio padrão) de perda e acurácia para múltiplas execuções.

    Parâmetros:
    - accuracies (list): Acurácias.
    - losses (list): Perdas.
    - model_name (str): Nome do modelo para exibição.

    Retorno:
    - summary (dict): Dicionário com estatísticas agregadas.
    """
    metrics = np.array(list(zip(losses, accuracies)))
    losses = metrics[:, 0]
    accuracies = metrics[:, 1]

    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)

    print(f"\n{model_name}:")
    print(f"  Acurácia média: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Perda média:    {loss_mean:.4f} ± {loss_std:.4f}")

    return {
        "model": model_name,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std
    }

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera um heatmap Grad-CAM para a imagem e modelo fornecidos.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def apply_gradcam(model, img_path, last_conv_layer_name='conv2d_2', input_size=(150, 150)):
    """
    Aplica Grad-CAM sobre uma imagem e exibe a sobreposição do heatmap.
    """
    # Carregar e processar imagem
    img = tf.keras.preprocessing.image.load_img(
        img_path, color_mode='grayscale', target_size=input_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Garantir shape (1, H, W, 1)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Gerar heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Plotagem
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array[0, :, :, 0], cmap='gray')
    plt.title("Imagem Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_array[0, :, :, 0], cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_last_conv_layer_name(model):
    """
    Retorna o nome da última camada convolucional 2D (Conv2D) do modelo.

    Parâmetros:
    - model: Um modelo Keras já construído.

    Retorno:
    - str: Nome da última camada Conv2D.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("O modelo não possui nenhuma camada Conv2D.")