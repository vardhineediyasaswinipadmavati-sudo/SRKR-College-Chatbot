import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config
from data_preprocessing import SRKRDataPreprocessor, run_pipeline

# Ensure reproducibility
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

def build_simplernn_model(vocab_size: int, num_classes: int, max_len: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=config.EMBEDDING_DIM),
        layers.SimpleRNN(config.RECURRENT_UNITS, return_sequences=False),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.DENSE_UNITS, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(num_classes, activation="softmax")
    ], name="SimpleRNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_lstm_model(vocab_size: int, num_classes: int, max_len: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=config.EMBEDDING_DIM),
        layers.LSTM(config.RECURRENT_UNITS, return_sequences=False),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.DENSE_UNITS, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(num_classes, activation="softmax")
    ], name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_gru_model(vocab_size: int, num_classes: int, max_len: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=config.EMBEDDING_DIM),
        layers.GRU(config.RECURRENT_UNITS, return_sequences=False),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.DENSE_UNITS, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(num_classes, activation="softmax")
    ], name="GRU")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_bilstm_model(vocab_size: int, num_classes: int, max_len: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=config.EMBEDDING_DIM),
        layers.Bidirectional(layers.LSTM(config.RECURRENT_UNITS // 2, return_sequences=False)),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(config.DENSE_UNITS, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE),
        layers.Dense(num_classes, activation="softmax")
    ], name="BiLSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def measure_dl_inference_time(model: keras.Model, sample_input: np.ndarray, n_trials: int = 50) -> float:
    """Measure average inference time per query in milliseconds."""
    # Warm up
    _ = model.predict(sample_input[:1], verbose=0)
    start_time = time.perf_counter()
    for _ in range(n_trials):
        _ = model.predict(sample_input[:1], verbose=0)
    end_time = time.perf_counter()
    return ((end_time - start_time) / n_trials) * 1000.0

def train_dl_model(
    name: str,
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    print(f"\n--- Training {name} ---")
    model_save_path = config.MODELS_DIR / f"{name.lower()}.keras"
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        )
    ]

    start_train = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.perf_counter() - start_train

    # Make sure best model is saved
    model.save(str(model_save_path))

    # Evaluate on Validation set (for fair model selection report)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    # Evaluate on Final Test set
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    inference_ms = measure_dl_inference_time(model, X_test)

    # Save history
    history_dict = {
        "loss": [float(v) for v in history.history["loss"]],
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "epochs_trained": len(history.history["loss"])
    }
    with open(config.METRICS_DIR / f"{name.lower()}_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    cm = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "model_name": name,
        "type": "Deep Learning",
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "train_time_sec": round(train_time, 3),
        "inference_time_ms": round(inference_ms, 3),
        "epochs_trained": history_dict["epochs_trained"],
        "confusion_matrix": cm,
        "y_true": [int(v) for v in y_test],
        "y_pred": [int(v) for v in y_pred]
    }

    print(f"{name} Completed: Test Acc={test_acc:.4f}, Val Acc={val_acc:.4f}, Macro F1={f1:.4f}, Train Time={train_time:.2f}s, Latency={inference_ms:.2f}ms")
    return result

def train_tfidf_baseline(
    patterns_train: list,
    y_train: np.ndarray,
    patterns_val: list,
    y_val: np.ndarray,
    patterns_test: list,
    y_test: np.ndarray
) -> Dict[str, Any]:
    print("\n--- Training TF-IDF + Logistic Regression Baseline ---")
    start_train = time.perf_counter()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(patterns_train)
    X_val_vec = vectorizer.transform(patterns_val)
    X_test_vec = vectorizer.transform(patterns_test)

    clf = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    clf.fit(X_train_vec, y_train)
    train_time = time.perf_counter() - start_train

    # Save baseline pipeline
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, config.MODELS_DIR / "baseline_tfidf.joblib")

    # Validation evaluation
    y_val_pred = clf.predict(X_val_vec)
    val_acc = float(accuracy_score(y_val, y_val_pred))

    # Test evaluation
    y_pred = clf.predict(X_test_vec)
    test_acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # Measure latency
    start_inf = time.perf_counter()
    for _ in range(100):
        _ = clf.predict_proba(vectorizer.transform([patterns_test[0]]))
    inference_ms = ((time.perf_counter() - start_inf) / 100.0) * 1000.0

    cm = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "model_name": "TF-IDF + Logistic Regression",
        "type": "Baseline ML",
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "train_time_sec": round(train_time, 3),
        "inference_time_ms": round(inference_ms, 3),
        "epochs_trained": 0,
        "confusion_matrix": cm,
        "y_true": [int(v) for v in y_test],
        "y_pred": [int(v) for v in y_pred]
    }

    print(f"TF-IDF Baseline Completed: Test Acc={test_acc:.4f}, Val Acc={val_acc:.4f}, Macro F1={f1:.4f}, Train Time={train_time:.2f}s, Latency={inference_ms:.2f}ms")
    return result

def main():
    print("=" * 60)
    print("STARTING FULL MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Run Preprocessing
    preprocessor, splits = run_pipeline()
    vocab_size = len(preprocessor.word2idx)
    num_classes = len(preprocessor.tag2idx)
    max_len = preprocessor.max_len

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    patterns_train = splits["patterns_train"]
    patterns_val = splits["patterns_val"]
    patterns_test = splits["patterns_test"]

    all_results = []

    # 2. Train DL Models
    # Model 1: SimpleRNN
    rnn_model = build_simplernn_model(vocab_size, num_classes, max_len)
    rnn_res = train_dl_model("SimpleRNN", rnn_model, X_train, y_train, X_val, y_val, X_test, y_test)
    all_results.append(rnn_res)

    # Model 2: LSTM
    lstm_model = build_lstm_model(vocab_size, num_classes, max_len)
    lstm_res = train_dl_model("LSTM", lstm_model, X_train, y_train, X_val, y_val, X_test, y_test)
    all_results.append(lstm_res)

    # Model 3: GRU
    gru_model = build_gru_model(vocab_size, num_classes, max_len)
    gru_res = train_dl_model("GRU", gru_model, X_train, y_train, X_val, y_val, X_test, y_test)
    all_results.append(gru_res)

    # Model 4: Bidirectional LSTM
    bilstm_model = build_bilstm_model(vocab_size, num_classes, max_len)
    bilstm_res = train_dl_model("BiLSTM", bilstm_model, X_train, y_train, X_val, y_val, X_test, y_test)
    all_results.append(bilstm_res)

    # 3. Train Baseline
    baseline_res = train_tfidf_baseline(patterns_train, y_train, patterns_val, y_val, patterns_test, y_test)
    all_results.append(baseline_res)

    # 4. Save metrics
    with open(config.METRICS_DIR / "all_models_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
