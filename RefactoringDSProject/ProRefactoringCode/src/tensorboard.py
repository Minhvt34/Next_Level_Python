from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.tensorboard import SummaryWriter
from src.tracking import Stage



class TensorboardExperiment:

    stage: Stage

    def __init__(self, log_dir: str, create=True):
        self._validate_log_dir(log_dir, create=create)
        self._writer = SummaryWriter(log_dir=log_dir)
        plt.ioff()

    def set_stage(self, stage: Stage):
        self.stage = stage
        return self
    
    def flush(self):
        self._writer.flush()


    @staticmethod
    def _validate_log_dir(log_dir, create=True):
        log_dir = Path(log_dir).resolve()
        if log_dir.exists():
            return
        elif not log_dir.exists() and create:
            log_dir.mkdir(parents=True)
        else:
            raise NotADirectoryError(f'log_dir {log_dir} does not exist.')

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f'{self.stage.name}/batch/{name}'
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f'{self.stage.name}/epoch/{name}'
        self._writer.add_scalar(tag, value, step)

    def add_epoch_confusion_matrix(self, y_true: [np.array], y_pred: [np.array], step: int):
        y_true, y_pred = self.collapse_batches(y_true, y_pred)
        fig = self.create_confusion_matrix(y_true, y_pred, step)
        tag = f'{self.stage.name}/epoch/confusion_matrix'
        self._writer.add_figure(tag, fig, step)

    
    @staticmethod
    def collapse_batches(y_true: [np.array], y_pred: [np.array]) -> [np.array, np.array]:
        return np.concatenate(y_true), np.concatenate(y_pred)

    def create_confusion_matrix(self, y_true: np.array, y_pred: np.array, step: int) -> plt.figure:
        cm = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap='Blues')
        fig: plt.figure = cm.figure_
        ax: plt.axes = cm.ax_
        ax.set_title(f'{self.stage.name} Epoch: {step}')
        return fig

    def add_hparams(self, hparams: [str, Union[str, float]], metrics: [str, float]):
        _metric = self._validate_hparam_metric_keys(metrics)
        self._writer.add_hparams(hparams, _metric)

    @staticmethod
    def _validate_hparam_metric_keys(metrics):
        _metrics = metrics.copy()
        prefix = 'hparam/'
        for name in _metrics.keys():
            if not name.startswith(prefix):
                _metrics[f'{prefix}{name}'] = _metrics[name]
                del _metrics[name]
        return _metrics