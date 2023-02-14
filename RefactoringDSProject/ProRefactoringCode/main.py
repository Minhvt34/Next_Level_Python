import torch
import pathlib

from src.dataset import create_dataloader
from src.models import LinearNet
from src.tensorboard import TensorboardExperiment
from running import Runner, run_epoch

# Log configuration
LOG_PATH = "./runs"

# Hyperparameters
EPOCH_COUNT = 20
LR = 5e-5
BATCH_SIZE = 128

# Data configuration
DATA_DIR = "./data"
TEST_DATA = pathlib.Path(f"{DATA_DIR}/t10k-images-idx3-ubyte.gz")
TEST_LABELS = pathlib.Path(f"{DATA_DIR}/t10k-labels-idx1-ubyte.gz")
TRAIN_DATA = pathlib.Path(f"{DATA_DIR}/train-images-idx3-ubyte.gz")
TRAIN_LABELS = pathlib.Path(f"{DATA_DIR}/train-labels-idx1-ubyte.gz")

def main():
    # Data
    train_loader = create_dataloader(TRAIN_DATA, TRAIN_LABELS, BATCH_SIZE)
    test_loader = create_dataloader(TEST_DATA, TEST_LABELS, BATCH_SIZE)

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    tracker = TensorboardExperiment(log_dir=LOG_PATH)

    # Run the epochs
    for epoch_id in range(EPOCH_COUNT):
        run_epoch(test_runner, train_runner, tracker, epoch_id)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{EPOCH_COUNT}]",
                f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )

        print("\n" + summary + "\n")

        # Reset the runner
        train_runner.reset()
        test_runner.reset()

        tracker.flush()


if __name__ == "__main__":
    main()