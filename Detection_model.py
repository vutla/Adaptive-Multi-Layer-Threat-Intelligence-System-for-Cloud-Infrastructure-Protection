import tensorflow as tf
from tensorflow.keras import layers, models, Input
from confusion_met import *
import time
def build_lstm_cnn(input_shape):
    inp = Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(16)(x)

    x = layers.Reshape((x.shape[1], 1))(x)
    x = layers.Conv1D(32, 3, activation='relu')(x)
    x = layers.Conv1D(16, 3, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    out = layers.Dense(32, activation='relu')(x)

    return models.Model(inp, out, name="LSTM_CNN")


def build_gnn_like(input_shape):
    inp = Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)  #  Fix shape: (None, 32)
    return models.Model(inp, x, name="GNN_Like")

def build_transformer(input_shape):
    inp = Input(shape=input_shape)  # shape: (36, 1)

    # Project input to higher dim if needed
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(inp)

    # Apply Transformer logic
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)

    return models.Model(inp, x, name="Transformer")

def proposed(X_train, X_test, y_train, y_test):
    # Ensure input has 3D shape for LSTM
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    input_shape = X_train.shape[1:]

    # Sub-models
    lstm_cnn = build_lstm_cnn(input_shape)
    gnn_like = build_gnn_like(input_shape)
    transformer = build_transformer(input_shape)

    # Inputs and concatenation
    inp = Input(shape=input_shape)
    out1 = lstm_cnn(inp)
    out2 = gnn_like(inp)
    out3 = transformer(inp)

    merged = layers.Concatenate()([out1, out2, out3])
    merged = layers.Dense(64, activation='relu')(merged)
    merged = layers.Dense(len(np.unique(y_train)), activation='softmax')(merged)

    model = models.Model(inp, merged)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=1)

    # === Measure Response Latency ===
    start_time = time.time()
    ypred = model.predict(X_test)
    end_time = time.time()

    total_time = end_time - start_time  # in seconds
    latency_ms = (total_time / len(X_test)) * 1000  # ms per sample
    print(f"Response Latency: {latency_ms:.2f} ms per sample")

    ypred_labels = np.argmax(ypred, axis=1)
    met = confu_matrix(y_test, ypred_labels)

    return met, latency_ms

