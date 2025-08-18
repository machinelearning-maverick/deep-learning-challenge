from deep_learning_challenge.machine_learning.deep_learning.challenge.day_23.cnn_with_pytorch import (
    DATASET,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    DEVICE,
    SimpleCNN,
    prepare_train_test_data_loader,
    prepare_model_optimizer_criterion,
    train_one_epoch,
    evaluate,
)


def test_train_for_epochs():
    # DATASET -> "mnist":
    in_channels = 1
    num_classes = 10

    train_loader, test_loader = prepare_train_test_data_loader()
    model, optimizer, criterion = prepare_model_optimizer_criterion(
        in_channels, num_classes, DEVICE, LR, WEIGHT_DECAY
    )

    print(f"Training on {DATASET.upper} | device={DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        validation_loss, validation_accuracy = evaluate(
            model, test_loader, criterion, DEVICE
        )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} || "
        f"Val Loss: {validation_loss:.4f} | Val Acc: {validation_accuracy:.4f}"
    )
