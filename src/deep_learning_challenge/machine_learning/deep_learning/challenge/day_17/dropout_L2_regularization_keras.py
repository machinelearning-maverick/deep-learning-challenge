from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=3,
    n_informative=15,
    n_redundant=0,
    random_state=42,
)

# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# one-hot encoded labels for Keras
y_train_oh = to_categorical(y_train.numpy())
y_test_oh = to_categorical(y_test.numpy())

# Keras model
model = Sequential(
    [
        Dense(
            64,
            input_shape(
                20,
            ),
            activation="relu",
            kernel_regularizes=l2(1e-4),
        ),
        Dropout(0.5),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ]
)
