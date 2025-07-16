import numpy as np

from deep_learning_challenge.machine_learning.deep_learning.challenge.day_07.loss_functions_from_scratch_numpy_mse_bce import (
    mse,
)


def test_mse():
    # temperature predictions
    y_true = np.array([15.5, 16.1, 17.3, 18.0])

    # temperature good enough predictions
    y_pred_good = np.array([15.0, 15.6, 17.1, 17.6])
    # temperature bad predictions
    y_pred_bad = np.array([12.5, 14.1, 12.1, 11.6])

    mse_small = mse(y_true, y_pred_good)
    mse_large = mse(y_true, y_pred_bad)

    assert mse_small < 0.2
    assert mse_large > 12
    assert mse_small < mse_large, "Good predictions should yield lower loss"
