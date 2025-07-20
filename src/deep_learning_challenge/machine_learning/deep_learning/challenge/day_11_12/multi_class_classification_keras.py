from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import make_classification

# Create synthetic 3-class classification data
X, y = make_classification(
    n_classes=3,
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# Convert y to categorical
y_cat = to_categorical(y)

# Model
model = Sequential([Dense(3, input_shape=(2,), activation="softmax")])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(X, y_cat, epochs=200, batch_size=32, verbose=0)
loss, acc = model.evaluate(X, y_cat, verbose=0)
print(f"Keras model – Loss: {loss:.4f} | Accuracy: {acc:.4f}")
