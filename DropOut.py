import tensorflow as tf
import tensorflow.keras.layers as tk_layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import os
import json
import pandas as pd


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


class ModelDropOut:
    def __init__(self, n_model, epochs, path_train, path_val, b_dim, aug=0):
        self.shape = (64, 64, 3)
        self.n_classes = 3
        self.name_model = n_model
        self.history = n_model[:n_model.index('.h5')] + '.txt'
        self.epochs = int(epochs)
        self.batch_dim_train = int(b_dim)
        self.batch_dim_val = int(b_dim)
        self.n_image_train = len(glob.glob(os.path.join(path_train, '*/*.png')))
        self.n_image_val = len(glob.glob(os.path.join(path_val, '*/*.png')))

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
        iups = MyCallback(progress_callback, view, self.epochs)
        # checkpoint,
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

        hist_df = pd.DataFrame(history.history)

        # save to json:
        with open(self.history, mode='w') as f:
            hist_df.to_json(f)


if __name__ == '__main__':

    Obj_model = ModelDropOut(epochs=100, path_train='C:/Users/piero/test2', path_val='C:/Users/piero/test2')
    Model_1, history = Obj_model.start_train()

    #name = 'Model_1_t' + '.h5'

    #Model_1.save(name)
