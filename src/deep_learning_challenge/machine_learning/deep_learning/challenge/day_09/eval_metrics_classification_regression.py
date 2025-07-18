import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_metrics_classification():
    # Ground truth and predictions
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred_good = np.array([1, 0, 1, 1, 0, 1, 0, 0])  # perfect

    # Compute metrics
    print(f"Accuracy (good): {accuracy_score(y_true, y_pred_good)}")
    print(f"Precision (good): {precision_score(y_true, y_pred_good)}")
    print(f"Recall (good): {recall_score(y_true, y_pred_good)}")
    print(f"F1 Score (good): {f1_score(y_true, y_pred_good)}")
    print(f"Confusion Matrix (good):\n {confusion_matrix(y_true, y_pred_good)}")


def eval_metrics_regression():

    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_good = np.array([2.5, 0.0, 2.1, 7.8])
    y_pred_bad = np.array([0.0, 0.0, 0.0, 0.0])

    print("MSE (good):", mean_squared_error(y_true, y_pred_good))
    print("MAE (good):", mean_absolute_error(y_true, y_pred_good))
    print("R^2 (good):", r2_score(y_true, y_pred_good))

    print("MSE (bad):", mean_squared_error(y_true, y_pred_bad))
    print("MAE (bad):", mean_absolute_error(y_true, y_pred_bad))
    print("R^2 (bad):", r2_score(y_true, y_pred_bad))
