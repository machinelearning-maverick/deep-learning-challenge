import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data():
    # Data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=0,
        random_state=42,
    )

    # Train & Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=3)

    return X_train, X_val, y_train_oh, y_val_oh


def prepare_datasets(X_train, X_val, y_train_oh, y_val_oh):
    # Prepare dataset as tf.data.Dataset
    buffer_size = 1000
    batch_size = 32

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_oh)).batch(
        batch_size
    )

    return train_dataset, val_dataset


def create_model():
    model = models.Sequential(
        [
            layers.Input(shape=(20,)),
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(32),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dense(3, activation="softmax"),
        ]
    )
    return model


def training_loop(model, optimizer, train_dataset, val_dataset, epochs=20):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y_batch, logits)

        train_acc = train_acc_metric.result()
        print(f"Train acc: {train_acc:.4f}")
        train_acc_metric.reset_state()

        # Validation
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_acc = val_acc_metric.result()
        print(f"Validation accuracy: {val_acc:.4f}")
        val_acc_metric.reset_state()

    return train_acc, val_acc


def plot():
    pass
