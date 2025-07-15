from deep_learning_challenge.machine_learning.deep_learning.challenge.day_06.from_tensor_to_model_sdf_pytorch import (
    pytorch_shape_debug,
)


def test_pytorch_shape_debug():
    shapes = pytorch_shape_debug()

    assert shapes[0] == (4, 3)
    assert shapes[1] == (4, 5)
    assert shapes[2] == (4, 5)
    assert shapes[3] == (4, 2)
