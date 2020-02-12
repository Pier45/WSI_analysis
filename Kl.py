import tensorflow as tf
import tensorflow.keras.layers as tk_layer
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import tensorflow_probability as tfp
import glob
import os
import json

class MyCallback(Callback):
    def __init__(self, progress, view, tot):
        self.view = view
        self.progress = progress
        self.tot = tot
        self.logs = {}

    def on_batch_end(self, batch, logs={}):
        self.logs = logs
        self.view.emit('===> Batch: {:5}   Accuracy: {:5.3f}'.format(batch, logs['acc']))

    def on_epoch_end(self, epoch, logs={}):
        self.view.emit('Epoch:  {:5}   Loss:  {:13.2f}   --   Train acc:  {:5.3f}   '
                       'Val acc:  {:5.3f}'.format(int(epoch)+1, logs['loss'], logs['acc'], logs['val_acc']))
        self.progress.emit(100*(int(epoch)+1)/self.tot)


class ModelKl:

    def __init__(self, n_model, epochs, path_train, path_val, b_dim, aug=0):
        self.shape = (64, 64, 3)
        self.n_classes = 3
        self.epochs = int(epochs)
        self.name_model = n_model
        self.history = n_model[:n_model.index('.h5')] + '.txt'
        self.batch_dim_train = int(b_dim)
        self.batch_dim_val = int(b_dim)
        self.n_image_train = len(glob.glob(os.path.join(path_train, '*/*.png')))
        self.n_image_val = len(glob.glob(os.path.join(path_val, '*/*.png')))
        self.tfd = tfp.distributions

        self.path_train = path_train
        self.path_val = path_val
        self.aug = aug

    def load_train(self):
        if self.aug == 1:
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               brightness_range=(0.5, 1),
                                               horizontal_flip=True,
                                               fill_mode="nearest"
                                               )
        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255)

        return train_datagen

    def flow_directory(self):
        train_datagen = self.load_train()

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.path_train,
            target_size=(64, 64),
            batch_size=self.batch_dim_train,
            classes=['AC', 'H', 'AD'])

        validation_generator = test_datagen.flow_from_directory(
            self.path_val,
            target_size=(64, 64),
            batch_size=self.batch_dim_val,
            classes=['AC', 'H', 'AD'])

        return train_generator, validation_generator

    def bayesian_vgg(self, input_shape,
                     num_classes=3,
                     kernel_posterior_scale_mean=-9.0,
                     kernel_posterior_scale_stddev=0.1,
                     kernel_posterior_scale_constraint=0.2):
        """Constructs a VGG16 model.
        Args:
          input_shape: A `tuple` indicating the Tensor shape.
          num_classes: `int` representing the number of class labels.
          kernel_posterior_scale_mean: Python `int` number for the kernel
            posterior's scale (log variance) mean. The smaller the mean the closer
            is the initialization to a deterministic network.
          kernel_posterior_scale_stddev: Python `float` number for the initial kernel
            posterior's scale stddev.
            ```
            q(W|x) ~ N(mu, var),
            log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
            ````
          kernel_posterior_scale_constraint: Python `float` number for the log value
            to constrain the log variance throughout training.
            i.e. log_var <= log(kernel_posterior_scale_constraint).
        Returns:
          tf.keras.Model.
        """

        filters = [16, 32, 128, 128, 200, 256]
        kernels = [16, 8, 8, 4, 4, 3]
        strides = [2, 1, 2, 1, 2, 2]
        maxp = [2, 2, 2, 2, 2, 2]

        def _untransformed_scale_constraint(t):
          return tf.clip_by_value(t, -1000, tf.math.log(kernel_posterior_scale_constraint))

        kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
            untransformed_scale_initializer=tf.compat.v1.initializers.random_normal(
                mean=kernel_posterior_scale_mean,
                stddev=kernel_posterior_scale_stddev),
            untransformed_scale_constraint=_untransformed_scale_constraint)

        image = tk_layer.Input(shape=input_shape)

        x = image
        for i in range(len(kernels)):
          x = self._vggconv_block(
              x,
              filters[i],
              kernels[i],
              strides[i],
              kernel_posterior_fn,
              maxp[i])

        x = tk_layer.Flatten()(x)
        x = tfp.layers.DenseFlipout(256, kernel_posterior_fn=kernel_posterior_fn)(x)
        x = tfp.layers.DenseFlipout(num_classes, kernel_posterior_fn=kernel_posterior_fn)(x)
        model = tf.keras.Model(inputs=image, outputs=x, name='vgg16')
        return model


    def _vggconv_block(self, x, filters, kernel, stride, kernel_posterior_fn, maxp):
        """Network block for VGG."""
        out = tfp.layers.Convolution2DFlipout(
            filters,
            kernel,
            padding='same',
            kernel_posterior_fn=kernel_posterior_fn)(x)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation('relu')(out)

        out = tfp.layers.Convolution2DFlipout(
            filters,
            kernel,
            padding='same',
            kernel_posterior_fn=kernel_posterior_fn)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation('relu')(out)
        if maxp!=0:
            out = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=stride)(out)
        return out

    def start_train(self, progress_callback, view):
        model = self.bayesian_vgg(input_shape=self.shape)
        model.summary()
        model.compile(loss='kullback_leibler_divergence', optimizer='Adadelta' , metrics=['accuracy'])

        train, val = self.flow_directory()

        STEPS_PER_EPOCH_T = self.n_image_train // self.batch_dim_train
        STEPS_PER_EPOCH_V = self.n_image_val // self.batch_dim_val
        filepath = "weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlyStopping = EarlyStopping(monitor='val_acc', patience=25, verbose=0, mode='max')
        iups = MyCallback(progress_callback, view, self.epochs)

        callbacks_list = [earlyStopping, iups]

        history = model.fit_generator(train,
                                      steps_per_epoch=STEPS_PER_EPOCH_T,
                                      epochs=self.epochs,
                                      verbose=1,
                                      validation_data=val,
                                      validation_steps=STEPS_PER_EPOCH_V,
                                      callbacks=callbacks_list
                                      )
        model.save(self.name_model)
        with open(self.history, 'w') as file:
            json.dump(history.history, file)
        #return model, history


if __name__ == '__main__':
    Obj_model = ModelKl()
    Model_1, history = Obj_model.start_train()

    name = 'Model_KL_52_52' + '.h5'

    Model_1.save(name)