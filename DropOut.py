"""
Bayesian dropout CNN for histological-tile classification (AC / AD / H).

Monte Carlo Dropout is implemented by keeping ``training=True`` in every
Dropout layer so that dropout is active at *inference* time as well.
Running the forward pass N times on the same input yields a distribution
over predictions, from which aleatoric and epistemic uncertainty can be
estimated.

Usage (standalone):
    python model_training.py
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)

# from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
# from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)
N_CLASSES: int = 3
CLASS_NAMES = ["AC", "H", "AD"]

# Each ConvBlock is described by (n_filters, kernel_size, use_pooling, dropout_rate).
# Setting use_pooling=False replaces the MaxPooling layer with a stride-1 identity
# (the old sentinel value of stride=-1).
@dataclass(frozen=True)
class ConvBlockConfig:
    filters: int
    kernel_size: int
    use_pooling: bool
    dropout_rate: float


CONV_BLOCKS: Tuple[ConvBlockConfig, ...] = (
    ConvBlockConfig(filters=16,  kernel_size=6, use_pooling=True,  dropout_rate=0.15),
    ConvBlockConfig(filters=32,  kernel_size=6, use_pooling=True,  dropout_rate=0.25),
    ConvBlockConfig(filters=64,  kernel_size=6, use_pooling=True,  dropout_rate=0.25),
    ConvBlockConfig(filters=128, kernel_size=4, use_pooling=True,  dropout_rate=0.25),
    ConvBlockConfig(filters=256, kernel_size=4, use_pooling=False, dropout_rate=0.30),
)

# ---------------------------------------------------------------------------
# Keras callback
# ---------------------------------------------------------------------------


class TrainingProgressCallback(Callback):
    """
    Emits per-batch and per-epoch training metrics via Qt signals.

    Parameters
    ----------
    progress_signal:
        PyQt signal that accepts an ``int`` (0–100 percentage).
    view_signal:
        PyQt signal that accepts a ``str`` status message.
    total_epochs:
        Total number of training epochs; used to compute the progress percentage.
    """

    def __init__(self, progress_signal, view_signal, total_epochs: int) -> None:
        super().__init__()
        self._progress = progress_signal
        self._view = view_signal
        self._total_epochs = total_epochs

    def on_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        acc = logs.get("accuracy", float("nan"))
        self._view.emit(f"===> Batch: {batch:5d}   Accuracy: {acc:5.3f}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        loss = logs.get("loss", float("nan"))
        acc = logs.get("accuracy", float("nan"))
        val_acc = logs.get("val_accuracy", float("nan"))
        epoch_number = int(epoch) + 1

        self._view.emit(
            f"Epoch: {epoch_number:5d}   "
            f"Loss: {loss:13.2f}   "
            f"Train acc: {acc:5.3f}   "
            f"Val acc: {val_acc:5.3f}"
        )
        self._progress.emit(int(100 * epoch_number / self._total_epochs))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BayesianDropoutCNN:
    """
    Convolutional neural network with Monte Carlo Dropout for Bayesian
    uncertainty estimation on 64×64 RGB histological tiles.

    Parameters
    ----------
    model_save_path:
        File path where the trained model will be saved (e.g. ``"model.h5"``).
    epochs:
        Number of training epochs.
    path_train:
        Root directory of the training set; must contain one sub-folder per class.
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
        self.model_save_path = model_save_path
        self.history_save_path = model_save_path.replace(".h5", "_history.json")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.path_train = path_train
        self.path_val = path_val
        self.augment = augment

        self.n_train_images = len(glob.glob(os.path.join(path_train, "*/*.png")))
        self.n_val_images = len(glob.glob(os.path.join(path_val, "*/*.png")))
        logger.info(
            "Dataset — train: %d images, val: %d images",
            self.n_train_images,
            self.n_val_images,
        )

    # ------------------------------------------------------------------
    # Data generators
    # ------------------------------------------------------------------

    def _build_train_generator(self) -> ImageDataGenerator:
        """Return the training :class:`ImageDataGenerator`, with optional augmentation."""
        if self.augment:
            return ImageDataGenerator(
                rescale=1.0 / 255,
                shear_range=0.2,
                zoom_range=0.2,
                brightness_range=(0.5, 1.0),
                horizontal_flip=True,
                fill_mode="nearest",
            )
        return ImageDataGenerator(rescale=1.0 / 255)

    def _build_data_generators(self):
        """
        Create and return ``(train_generator, validation_generator)``.

        Both generators yield batches of shape ``(batch_size, 64, 64, 3)``
        with one-hot encoded labels for the three classes.
        """
        train_datagen = self._build_train_generator()
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        flow_kwargs = dict(
            target_size=INPUT_SHAPE[:2],
            batch_size=self.batch_size,
            classes=CLASS_NAMES,
        )

        train_gen = train_datagen.flow_from_directory(self.path_train, **flow_kwargs)
        val_gen = val_datagen.flow_from_directory(self.path_val, **flow_kwargs)
        return train_gen, val_gen

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_block(x: tf.Tensor, cfg: ConvBlockConfig) -> tf.Tensor:
        """
        Build a double-convolution block with BatchNorm, optional pooling,
        and Monte Carlo Dropout.

        Note: ``Conv2D`` layers use *no* activation; ``BatchNormalization``
        is followed by an explicit ``Activation('relu')`` to follow the
        canonical BN → ReLU ordering.

        The ``training=True`` flag on ``Dropout`` keeps it active during
        inference, enabling Monte Carlo sampling.
        """
        conv_kwargs = dict(
            filters=cfg.filters,
            kernel_size=cfg.kernel_size,
            padding="same",
            use_bias=False,  # bias is redundant before BatchNorm
        )

        for _ in range(2):
            x = tf.keras.layers.Conv2D(**conv_kwargs)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

        if cfg.use_pooling:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

        # training=True → Dropout active at inference time (Monte Carlo Dropout)
        x = tf.keras.layers.Dropout(cfg.dropout_rate)(x, training=True)
        return x

    def build_model(self) -> tf.keras.Model:
        """
        Construct and return the Bayesian dropout CNN.

        Architecture
        ------------
        5 × ConvBlock  →  Conv2D(1024) head  →  Dense(1024 → 364 → 256)  →  Softmax(3)
        """
        inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_tiles")
        x = inputs

        for block_cfg in CONV_BLOCKS:
            x = self._conv_block(x, block_cfg)

        # Classification head
        x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.35)(x, training=True)
        x = tf.keras.layers.Dense(364, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x, training=True)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.20)(x, training=True)

        outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="class_probabilities")(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name="bayesian_dropout_cnn")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, progress_signal=None, view_signal=None) -> tf.keras.callbacks.History:
        """
        Build, compile, and train the model.

        Parameters
        ----------
        progress_signal:
            Optional PyQt signal (``int``) for reporting epoch progress.
        view_signal:
            Optional PyQt signal (``str``) for streaming log messages to a UI widget.

        Returns
        -------
        tf.keras.callbacks.History
            Keras history object containing per-epoch metrics.
        """
        model = self.build_model()
        model.summary(print_fn=logger.info)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adadelta(),
            metrics=["accuracy"],
        )

        train_gen, val_gen = self._build_data_generators()

        steps_per_epoch = math.ceil(self.n_train_images / self.batch_size)
        validation_steps = math.ceil(self.n_val_images / self.batch_size)

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath="weights_best.h5",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
        ]

        if progress_signal is not None and view_signal is not None:
            callbacks.append(
                TrainingProgressCallback(progress_signal, view_signal, self.epochs)
            )

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(self.model_save_path)
        logger.info("Model saved to: %s", self.model_save_path)

        hist_df = pd.DataFrame(history.history)
        with open(self.history_save_path, mode="w") as fp:
            hist_df.to_json(fp)
        logger.info("Training history saved to: %s", self.history_save_path)

        return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = BayesianDropoutCNN(
        model_save_path="bayesian_cnn.h5",
        epochs=100,
        path_train="data/train",
        path_val="data/val",
        batch_size=32,
        augment=True,
    )
    training_history = trainer.train()
    logger.info("Final val accuracy: %.4f", max(training_history.history["val_accuracy"]))