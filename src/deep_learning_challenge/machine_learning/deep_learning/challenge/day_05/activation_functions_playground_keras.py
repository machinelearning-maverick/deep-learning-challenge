import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Input values
x = np.linspace(-5, 5, 200)

# Activation functions
activations = {
    "ReLU": K.eval(K.relu(x)),
    "Sigmoid": K.eval(K.sigmoid(x)),
    "Tanh": K.eval(K.tanh(x)),
    "LeakyReLU": K.eval(K.relu(x, alpha=0.1)),  # aplha == negative slop
    "Softplus": K.eval(K.softplus(x)),
}

# Plot
plt.figure(figsize=(10, 6))
for name, y in activations.items():
    plt.plot(x, y, label=name)

plt.plot(x, x, label="x (raw input)", linestyle="--", color="gray")

plt.title("Activation Functions (Keras)")
plt.xlabel("x")
plt.ylabel("activation(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# plt.savefig("activation_functions_playground_keras.png")

plt.show()
