from confusion_met import *
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models, Input
from tensorflow.keras import backend as K
import time
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def rbf_activation(x):
    return K.exp(-K.square(x))  # Gaussian RBF
def build_rbfnn(input_dim):
    inp = Input(shape=(input_dim,))
    x = layers.Dense(64)(inp)
    x = layers.Activation(rbf_activation)(x)
    x = layers.Dense(32)(x)
    x = layers.Activation(rbf_activation)(x)
    out = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(10, activation='softmax')(out)  # Adjust for number of classes
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Combined RBFNN + RF with Response Latency ---
def rbfnn_rf(X_train, X_test, y_train, y_test):
    num_classes = len(np.unique(y_train))

    # Scale features for RBFNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- RBFNN ---
    rbfnn = build_rbfnn(X_train.shape[1])
    rbfnn.fit(X_train_scaled, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2)

    # --- Measure latency ---
    start_time = time.time()

    # RBFNN Predictions
    rbfnn_probs = rbfnn.predict(X_test_scaled)
    rbfnn_preds = np.argmax(rbfnn_probs, axis=1)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # Fusion strategy: Confidence-based
    rbf_conf = np.max(rbfnn_probs, axis=1)
    final_preds = []
    for i in range(len(y_test)):
        if rbfnn_preds[i] == rf_preds[i]:
            final_preds.append(rf_preds[i])
        else:
            final_preds.append(rbfnn_preds[i] if rbf_conf[i] > 0.7 else rf_preds[i])

    # --- Measure end time ---
    response_latency = (time.time() - start_time) * 1000  # in milliseconds
    avg_latency_per_sample = response_latency / len(y_test)
    met = multi_confu_matrix(y_test, final_preds)
    return met, response_latency

def build_dnn(input_dim, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def dnn(X_train, X_test, y_train, y_test):
    num_classes = len(np.unique(y_train))

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train DNN
    model = build_dnn(X_train.shape[1], num_classes)
    model.fit(X_train_scaled, y_train,
              epochs=100,
              batch_size=128,
              verbose=1,
              validation_split=0.2)

    # Predict with latency measurement
    start_time = time.time()
    y_probs = model.predict(X_test_scaled)
    response_latency = time.time() - start_time
    avg_latency_per_sample = response_latency / len(X_test)
    # Final predictions
    y_pred = np.argmax(y_probs, axis=1)
    # Evaluation
    met = multi_confu_matrix(y_test, y_pred)
    return met, response_latency



def build_cnn(input_dim, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim, 1)))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def cnn(X_train, X_test, y_train, y_test):
    num_classes = len(np.unique(y_train))

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for Conv1D
    X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

    # Build and train CNN
    model = build_cnn(X_train.shape[1], num_classes)
    model.fit(X_train_cnn, y_train,
              epochs=100,
              batch_size=128,
              verbose=1,
              validation_split=0.2)

    # Predict with latency measurement
    start_time = time.time()
    y_probs = model.predict(X_test_cnn)
    response_latency = time.time() - start_time
    y_pred = np.argmax(y_probs, axis=1)
    # Evaluation
    met = multi_confu_matrix(y_test, y_pred)
    avg_latency_per_sample = response_latency / len(X_test)
    return met, response_latency



def svm_ga(X_train, X_test, y_train, y_test):
    # Handle missing values (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Build SVM (RBF kernel as default)
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

    # Train SVM
    model.fit(X_train_scaled, y_train)

    # Measure response latency (prediction time)
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    response_latency = (time.time() - start_time) * 1000  # in milliseconds
    # Evaluation
    met = multi_confu_matrix(y_test, y_pred)
    return met, response_latency

