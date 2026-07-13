"""
Shared interface for the Bayesian uncertainty models.

Both ``BayesianDropoutCNN`` and ``ModelKl`` implement this protocol, so the
GUI / training pipeline can drive them with the same code path without
branching on kwargs.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from tensorflow.keras.callbacks import History


@runtime_checkable
class BayesianModel(Protocol):
    """Construct a model for training and uncertainty estimation.

    Parameters (constructor)
    -------------------------
    model_save_path:
        File path where the trained model will be saved (e.g. ``"model.h5"``).
    epochs:
        Number of training epochs.
    path_train:
        Root directory of the training set; one sub-folder per class.
    path_val:
        Root directory of the validation set; same structure as *path_train*.
    batch_size:
        Batch size used for both training and validation generators.
    augment:
        Whether to apply data augmentation during training.
    """

    def __init__(
        self,
        model_save_path: str,
        epochs: int,
        path_train: str,
        path_val: str,
        batch_size: int = 32,
        augment: bool = False,
    ) -> None:
        ...

    def train(
        self,
        progress_signal: Optional[object] = None,
        view_signal: Optional[object] = None,
    ) -> History:
        """Build, compile and train the model, returning the Keras history."""
        ...

    def start_train(
        self,
        progress_callback: Optional[object] = None,
        view: Optional[object] = None,
    ) -> History:
        """Backward-compatible alias for :meth:`train`."""
        ...
