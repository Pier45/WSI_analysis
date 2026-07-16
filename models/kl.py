"""
Variational inference (KL-divergence) Bayesian CNN for histological-tile
classification (AC / AD / H).

The model uses TensorFlow-Probability Flipout layers
(``Convolution2DFlipout``, ``DenseFlipout``) to learn an approximate
posterior over the network weights. The KL-divergence term is automatically
added to the loss by each Flipout layer when ``model.fit`` is called, so no
explicit KL term is required in the loss function spec — but we use
``kullback_leibler_divergence`` as the data-fit loss term on the output
distribution, as in the original implementation.

Usage (standalone):
    python -m models.kl
"""

from __future__ import annotations

import glob
import logging
import math
import os

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture constants — single source of truth in src/config.py
# ---------------------------------------------------------------------------

from src.config import CLASS_NAMES, INPUT_SHAPE, N_CLASSES  # noqa: E402


class TrainingProgressCallback(Callback):
    """
    Emits per-batch and per-epoch training metrics via Qt signals.

    Parameters
    ----------
    progress_signal:
        PyQt signal that accepts an ``int`` (0-100 percentage).
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

    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        acc = logs.get("accuracy", float("nan"))
        self._view.emit(f"===> Batch: {batch:5d}   Accuracy: {acc:5.3f}")

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
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


class ModelKl:
    """
    Bayesian CNN using KL-divergence variational inference (TF-Probability
    Flipout layers) for 64x64 RGB histological tiles.

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
        self.augment = bool(augment)

        self.shape = INPUT_SHAPE
        self.n_classes = N_CLASSES
        self.n_train_images = len(glob.glob(os.path.join(path_train, "*/*.png")))
        self.n_val_images = len(glob.glob(os.path.join(path_val, "*/*.png")))
        self.tfd = tfp.distributions

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

    def bayesian_vgg(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int = N_CLASSES,
        kernel_posterior_scale_mean: float = -9.0,
        kernel_posterior_scale_stddev: float = 0.1,
        kernel_posterior_scale_constraint: float = 0.2,
    ) -> tf.keras.Model:
        """Constructs a Flipout Bayesian VGG-style model.

        Args:
            input_shape: A ``tuple`` indicating the Tensor shape.
            num_classes: ``int`` representing the number of class labels.
            kernel_posterior_scale_mean: Python ``float`` for the kernel
                posterior's scale (log variance) mean. The smaller the mean
                the closer the initialization is to a deterministic network.
            kernel_posterior_scale_stddev: Python ``float`` for the initial
                kernel posterior's scale stddev::

                    q(W|x) ~ N(mu, var),
                    log_var ~ N(kernel_posterior_scale_mean,
                                kernel_posterior_scale_stddev)

            kernel_posterior_scale_constraint: ``float`` constraining the
                log variance throughout training::

                    log_var <= log(kernel_posterior_scale_constraint)

        Returns:
            ``tf.keras.Model``.
        """

        filters = [16, 32, 128, 128, 200, 256]
        kernels = [16, 8, 8, 4, 4, 3]
        strides = [2, 1, 2, 1, 2, 2]
        maxp = [2, 2, 2, 2, 2, 2]

        def _untransformed_scale_constraint(t):
            return tf.clip_by_value(
                t, -1000, tf.math.log(kernel_posterior_scale_constraint)
            )

        kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
            untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
                mean=kernel_posterior_scale_mean,
                stddev=kernel_posterior_scale_stddev,
            ),
            untransformed_scale_constraint=_untransformed_scale_constraint,
        )

        image = tf.keras.layers.Input(shape=input_shape)

        x = image
        for i in range(len(kernels)):
            x = self._vggconv_block(
                x,
                filters[i],
                kernels[i],
                strides[i],
                kernel_posterior_fn,
                maxp[i],
            )

        x = tf.keras.layers.Flatten()(x)
        x = tfp.layers.DenseFlipout(256, kernel_posterior_fn=kernel_posterior_fn)(x)
        x = tfp.layers.DenseFlipout(num_classes, kernel_posterior_fn=kernel_posterior_fn)(x)
        model = tf.keras.Model(inputs=image, outputs=x, name="bayesian_kl_vgg")
        return model

    def _vggconv_block(self, x, filters, kernel, stride, kernel_posterior_fn, maxp):
        """Network block for VGG."""
        out = tfp.layers.Convolution2DFlipout(
            filters,
            kernel,
            padding="same",
            kernel_posterior_fn=kernel_posterior_fn,
        )(x)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation("relu")(out)

        out = tfp.layers.Convolution2DFlipout(
            filters,
            kernel,
            padding="same",
            kernel_posterior_fn=kernel_posterior_fn,
        )(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation("relu")(out)
        if maxp != 0:
            out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=stride)(out)
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, progress_signal=None, view_signal=None
    ) -> tf.keras.callbacks.History:
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
        model = self.bayesian_vgg(input_shape=self.shape)
        model.summary(print_fn=logger.info)

        model.compile(
            loss="kullback_leibler_divergence",
            optimizer="Adadelta",
            metrics=["accuracy"],
        )

        train_gen, val_gen = self._build_data_generators()

        steps_per_epoch = math.ceil(self.n_train_images / self.batch_size)
        validation_steps = math.ceil(self.n_val_images / self.batch_size)

        checkpoint_path = self.model_save_path.replace(".h5", "_best.h5")
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=25,
                mode="max",
                restore_best_weights=True,
                verbose=0,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
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

    # Backward-compatible alias for the GUI / older callers that invoked
    # ``start_train(progress_callback, view)``.
    def start_train(self, progress_callback=None, view=None) -> tf.keras.callbacks.History:
        return self.train(progress_signal=progress_callback, view_signal=view)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = ModelKl(
        model_save_path="bayesian_kl.h5",
        epochs=100,
        path_train="data/train",
        path_val="data/val",
        batch_size=32,
        augment=True,
    )
    training_history = trainer.train()
    logger.info(
        "Final val accuracy: %.4f",
        max(training_history.history["val_accuracy"]),
    )
