from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.initializers import HeNormal, GlorotUniform, RandomUniform

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_15.weight_initialization_strategies_keras import (
    model_with_initialization,
    plot_impact_on_training,
)


MODELS_INIT_DATA = []


def test_model_with_initialization():
    # Prepare data
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)

    history_he_gu, loss_he_gu, accuracy_he_gu = model_with_initialization(
        X, y, HeNormal(), GlorotUniform()
    )
    MODELS_INIT_DATA.append(
        {
            "initializers": "HeNormal + GlorotUniform",
            "history": history_he_gu,
            "loss": loss_he_gu,
            "accuracy": accuracy_he_gu,
        }
    )

    history_ra_un, loss_ra_un, accuracy_ra_un = model_with_initialization(
        X,
        y,
        RandomUniform(minval=-0.05, maxval=0.05),
        RandomUniform(minval=-0.05, maxval=0.05),
    )
    MODELS_INIT_DATA.append(
        {
            "initializers": "2x RandomUniform(-0.05, 0.05)",
            "history": history_ra_un,
            "loss": loss_ra_un,
            "accuracy": accuracy_ra_un,
        }
    )

    plot_impact_on_training(MODELS_INIT_DATA)

    pass
