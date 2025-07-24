import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add


def build_simple_cnn(input_shape=(150, 150, 1), dropout_rate=0.5):
    """
    Constrói uma CNN simples com 3 blocos de convolução + pooling e duas camadas densas.

    Parâmetros:
    - input_shape (tuple): Formato da imagem de entrada.
    - dropout_rate (float): Taxa de dropout na camada densa final.

    Retorno:
    - model: Modelo Keras compilado.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_tuned_cnn(input_shape=(150, 150, 1), dropout_rate=0.5):
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
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_residual_cnn(input_shape=(150, 150, 1), dropout_rate=0.5):
    """
    Constrói uma CNN com blocos residuais usando a API funcional do Keras.

    Parâmetros:
    - input_shape (tuple): Formato da imagem de entrada.
    - dropout_rate (float): Taxa de dropout na camada densa final.

    Retorno:
    - model: Modelo Keras compilado.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Bloco inicial
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Bloco residual 1
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)  # <-- ajuste aqui
    x = Add()([x, residual])

    # Bloco residual 2
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)  # <-- ajuste aqui
    x = Add()([x, residual])

    # Camadas finais
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model