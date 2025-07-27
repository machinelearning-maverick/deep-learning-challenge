import numpy as np

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_16.optimizers_from_scratch import (
    SGDOptimizer,
    MomentumOptimizer,
    train_model,
    plot_impact_on_training,
)


def test_train_model_with_optimizers():
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    # one-hot encoded labels
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    optimizer_sgd = SGDOptimizer()
    loss_history_sgd, acc_history_sgd, W_sgd, b_sgd = train_model(
        X, y, y_onehot, optimizer_sgd
    )
    plot_impact_on_training(
        loss_history_sgd, acc_history_sgd, W_sgd, b_sgd, "SGDOptimizer"
    )

    optimizer_momentum = MomentumOptimizer()
    loss_history_mntm, acc_history_mntm, W_mntm, b_mntm = train_model(
        X, y, y_onehot, optimizer_momentum
    )
    plot_impact_on_training(
        loss_history_mntm, acc_history_mntm, W_mntm, b_mntm, "MomentumOptimizer"
    )

    pass
