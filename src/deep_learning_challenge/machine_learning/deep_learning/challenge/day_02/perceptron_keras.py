import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Model
model = Sequential()
model.add(Dense(1, input_dim=2, activation="sigmoid"))

# Compile
model.compile(optimizer="sgd", loss="binary_crossentropy")

# Train
model.fit(X, y, epochs=100, verbose=0)

# Predict
preds = model.predict(X)
print(f"Predictions: {np.round(preds).flatten()}")


def custom_training_loop_weight_update():
    # Data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [0], [0], [1]], dtype=np.float32)

    # Convert np.array into TensorFlow tensors
    X_tf = tf.convert_to_tensor(X)
    y_tf = tf.convert_to_tensor(y)

    # Model build
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(2,), activation="sigmoid")]
    )

    # Loss & Optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    # Training
    epochs = 100
    for epoch in range(epochs):
        with tf.GradientTape() as type:
            y_pred = model(X_tf, training=True)
            loss = loss_fn(y_tf, y_pred)

            model_train_vars = model.trainable_variables

            # Compute gradients
            gradients = type.gradient(loss, model_train_vars)

            # Update weights explicitly
            optimizer.apply_gradients(zip(gradients, model_train_vars))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.numpy():.4f}")

    # Final predictions
    preds_ = model(X_tf).numpy().flatten()
    print(f"Predictions: {preds_}")
