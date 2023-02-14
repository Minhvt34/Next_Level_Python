from pathlib import Path
from typing import Union
from typing import Protocol

import numpy as np
from enum import Enum, auto
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.tensorboard import SummaryWriter

class Stage(Enum):
    TRAIN = auto()
    TEST= auto()
    VAL= auto()


class ExperimentTracker(Protocol):

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(self, y_true: np.array, y_pred: np.array, step: int):
        """Implements logging a confusion matrix at epoch-level."""

    def add_hparams(self, hparams: [str, Union[str, float]], metrics: [str, float]):
        """Implements logging a confusion matrix at epoch-level."""

    def set_stage(self, stage: Stage):
        """Sets the stage of the experiment."""

    def flush(self):
        """Flushes the experiment."""
