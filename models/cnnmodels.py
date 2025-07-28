import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

def build_simple_cnn(dropout_rate=0.5, learning_rate=1e-3, input_shape=(224, 224, 3)):
    """
    Constrói uma CNN simples com 3 blocos de convolução + pooling e duas camadas densas.

    Parâmetros:
    - input_shape (tuple): Formato da imagem de entrada.
    - dropout_rate (float): Taxa de dropout na camada densa final.

    Retorno:
    - model: Modelo Keras compilado.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_tuned_cnn(dropout_rate=0.5, learning_rate=1e-3, input_shape=(224, 224, 3)):
    """
    Constrói uma CNN com camadas de BatchNormalization e profundidade aumentada.

    Parâmetros:
    - input_shape (tuple): Formato da imagem de entrada.
    - dropout_rate (float): Taxa de dropout na camada densa final.

    Retorno:
    - model: Modelo Keras compilado.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_model_vgg16(dropout_rate=0.5, learning_rate=1e-3, input_shape=(224,224,3)):
    """
    Constrói uma CNN baseada no modelo VGG16, aproveitando os pesos das camadas de convolução.

    Parâmetros:
    - input_shape (tuple): Formato da imagem de entrada.
    - dropout_rate (float): Taxa de dropout na camada densa final.

    Retorno:
    - model: Modelo Keras compilado.
    """
    # Ajustar o backbone VGG16 para imagens 32x32
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar as camadas convolucionais do VGG16
    vgg16.trainable = False

    # Construir o modelo final
    model_vgg = Sequential([
        vgg16,
        Flatten(),
        Dense(128, activation='relu'),  # Camada densa intermediária
        Dropout(dropout_rate),          # Dropout para regularização
        Dense(5, activation='softmax') # Camada de saída para 5 classes
    ])

    model_vgg.compile(optimizer=Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model_vgg