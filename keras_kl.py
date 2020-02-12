import tensorflow as tf
import tensorflow.keras.layers as tk_layer
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_probability as tfp


class KerasKl:
    def __init__(self, epochs=1000, path_train='/home/ppolicastro/images/TRAIN/', path_val='/home/ppolicastro/images/TEST/', patience=50):
        self.shape = (64, 64, 3)
        self.n_classes = 3
        self.patience = patience
        self.epochs = epochs
        self.batch_dim_train = 400
        self.batch_dim_val = 400
        self.n_image_train = 45000
        self.n_image_val = 19200
        self.tfd = tfp.distributions

        self.path_train = path_train
        self.path_val = path_val

    def flow_directory(self):
        train_datagen = ImageDataGenerator(rescale=1./255
                                            #shear_range=0.2,
                                            #zoom_range=0.2,
                                            #brightness_range=(0.5, 1),
                                            #horizontal_flip=True,
                                            #fill_mode="nearest"
                                           )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.path_train,
            target_size=(64, 64),
            batch_size=self.batch_dim_train,
            classes=['p_AC', 'p_H', 'p_AD'])

        validation_generator = test_datagen.flow_from_directory(
            self.path_val,
            target_size=(64, 64),
            batch_size=self.batch_dim_val,
            classes=['p_AC', 'p_H', 'p_AD'])

        return train_generator, validation_generator

    def bayesian_vgg(self, input_shape,
                     num_classes=3,
                     kernel_posterior_scale_mean=-9.0,
                     kernel_posterior_scale_stddev=0.1,
                     kernel_posterior_scale_constraint=0.2):

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
        #x = tfp.layers.DenseFlipout(1024, kernel_posterior_fn=kernel_posterior_fn)(x)
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

    def start_train(self):
        model = self.bayesian_vgg(input_shape=self.shape)
        model.summary()
        model.compile(loss='kullback_leibler_divergence', optimizer='Adadelta', metrics=['accuracy'])

        train, val = self.flow_directory()

        STEPS_PER_EPOCH_T = self.n_image_train // self.batch_dim_train
        STEPS_PER_EPOCH_V = self.n_image_val // self.batch_dim_val
        file_path = "weights_kl.hdf5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlyStopping = EarlyStopping(monitor='val_acc', patience=self.patience, verbose=0, mode='max')
        callbacks_list = [checkpoint, earlyStopping]

        history = model.fit_generator(train,
                            steps_per_epoch=STEPS_PER_EPOCH_T,
                            epochs=self.epochs,
                            verbose = 1,
                            validation_data = val,
                            validation_steps = STEPS_PER_EPOCH_V,
                            callbacks = callbacks_list
                            )

        with open('dizionario_storia.json', 'w') as file:
            json.dump(history.history, file)

        name = 'Model_KL' + '.h5'
        Model_1.save(name)

        return model, history


if __name__ == '__main__':
    Obj_model = KerasKl()
    Model_1, history = Obj_model.start_train()


