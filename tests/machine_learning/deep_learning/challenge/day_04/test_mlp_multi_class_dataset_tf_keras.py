from deep_learning_challenge.machine_learning.deep_learning.challenge.day_04.mlp_multi_class_dataset_tf_keras import (
    train_iris_mlp,
)


def test_iris_model_accuracy_above_treshold():
    accuracy = train_iris_mlp()
    assert accuracy > 0.8, f"Expected accuracy > 0.8 but got {accuracy:.4f}"
