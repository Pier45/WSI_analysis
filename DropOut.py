import tensorflow as tf
import tensorflow.keras.layers as tk_layer
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


class MyCallback(Callback):
    def __init__(self, view):
        self.view = view

    def on_batch_end(self, batch, logs={}):
        self.view.emit('The average loss for epoch {}'.format(batch))

    def on_epoch_end(self, epoch, logs=None):
        print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))
        self.view.emit('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))


class ModelDropOut:
    def __init__(self, epochs, path_train, path_val):
        self.shape = (64, 64, 3)
        self.n_classes = 3
        self.epochs = epochs
        self.batch_dim_train = 400
        self.batch_dim_val = 200
        self.n_image_train = 61904
        self.n_image_val = 19656

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
            classes=['AC', 'H', 'AD'])

        validation_generator = test_datagen.flow_from_directory(
            self.path_val,
            target_size=(64, 64),
            batch_size=self.batch_dim_val,
            classes=['AC', 'H', 'AD'])

        return train_generator, validation_generator

    def costruction_model(self):
        filters = [16, 32, 64, 128, 256]
        kernels = [6, 6, 6, 4, 4]
        strides = [2, 2, 2, 1, -1]
        drop = [0.15, 0.25, 0.25, 0.25, 0.3]
        image = tk_layer.Input(shape=self.shape)

        x = image
        for i in range(len(kernels)):
          x = self.middle_layers(x, filters[i], kernels[i], strides[i], drop[i])

        x = tk_layer.Conv2D(filters=1024, kernel_size=3, padding='same', activation = 'relu')(x)
        x = tk_layer.BatchNormalization()(x)
        x = tk_layer.Activation('relu')(x)
        x = tk_layer.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = tk_layer.Flatten()(x)
        x = tk_layer.Dense(1024, activation='relu')(x)
        x = tk_layer.Dropout(0.35)(x, training=True)
        x = tk_layer.Dense(364, activation='relu')(x)
        x = tk_layer.Dropout(0.25)(x, training=True)
        x = tk_layer.Dense(256, activation='relu')(x)
        out_m = tk_layer.Dense(self.n_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=image, outputs=out_m, name='sasa45')
        return model

    def middle_layers(self, x, nfil, kernel, stride, drop):
        out = tk_layer.Conv2D(filters=nfil, kernel_size=kernel, padding='same', activation = 'relu')(x)
        out = tk_layer.BatchNormalization()(out)
        out = tk_layer.Activation('relu')(out)

        out = tk_layer.Conv2D(filters=nfil, kernel_size=kernel, padding='same', activation = 'relu')(out)
        out = tk_layer.BatchNormalization()(out)
        out = tk_layer.Activation('relu')(out)

        if stride != -1:
            out = tk_layer.MaxPooling2D(pool_size=(2, 2), strides=stride)(out)
        out = tk_layer.Dropout(drop)(out, training=True)

        return out

    def start_est(self, progress_callback, view):
        i = 0
        while i < 10:
            view.emit('caio'+str(i))
            progress_callback.emit(i)
            time.sleep(1)
            i +=1

    def start_train(self, progress_callback, view):
        model = self.costruction_model()
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='Adadelta' ,metrics=['accuracy'])

        train, val = self.flow_directory()

        STEPS_PER_EPOCH_T = self.n_image_train // self.batch_dim_train
        STEPS_PER_EPOCH_V = self.n_image_val // self.batch_dim_val
        filepath = "weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlyStopping = EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='max')
        iups = MyCallback(view)
        callbacks_list = [checkpoint, earlyStopping, iups]

        history = model.fit_generator(train,
                                      steps_per_epoch=STEPS_PER_EPOCH_T,
                                      epochs=self.epochs,
                                      verbose=1,
                                      validation_data=val,
                                      validation_steps=STEPS_PER_EPOCH_V,
                                      callbacks=callbacks_list
                                      )

        #return model, history


if __name__ == '__main__':

    Obj_model = ModelDropOut(epochs=100, path_train='C:/Users/piero/test2', path_val='C:/Users/piero/test2')
    Model_1, history = Obj_model.start_train()

    #name = 'Model_1_t' + '.h5'

    #Model_1.save(name)
