import os
import PIL
import sys
import time
import threading
import glob
import numpy as np
from scipy import misc
import tensorflow as tf


class Classification:
    def __init__(self, path):
        self.path = path
        self.dictionary = {}
        self.list_image = []
        self.shape = (128, 128, 3)

    def analysis_folder(self, sel_folder):
        """ Analyze the selected folder, finding all the png files"""

        list_files = glob.glob(sel_folder)
        n_elements = len(list_files)
        return list_files, n_elements

    def select_folder(self):
        """
        Create the dict with the key the number of the tile, to each key correspond another sub-dict that as key as
        'image' and value the path to the image.  At the same time it's created a list of numpy arrays that contain
        the image vectors already padded.
        """
        self.dictionary, sub_dic = {}, {}
        folders = os.listdir(self.path)
        print(folders)
        for i in folders:
            sel_folder = self.path + str(i) + '/' + '*.png'
            list_files, n_elements = self.analysis_folder(sel_folder)
            for j in list_files:
                complete_name = j[list(j).index('\\') + 1:-4]
                partial = complete_name[list(complete_name).index('_') + 1:]
                n_tile = partial[:partial.index('_')]
                np_image = self.tile_control(j)
                self.list_image.append(np_image)
                sub_dic['image_path'] = j
                self.dictionary[n_tile] = sub_dic
            print('Selected Folder:   {:<40} Number of elements: {}'.format(sel_folder, n_elements))
        print(self.dictionary['77'])

    def tile_control(self, j):
        """ This method read the image, modify it as numpy array and at the end control if same padding is needed."""
        image = misc.imread(j)
        np_image = np.asarray(image / 255, dtype=float)
        shape_i = image.shape
        if shape_i != self.shape:
            np_image = np.pad(image, ((0, shape_i[0] - self.shape[0]), (0, shape_i[1] - self.shape[1]), (0, 0)), 'constant', constant_values=1)
        return np_image

    def classify(self):
        """Load the model and analyze the tile, the dictionary is updated with the predicted label
        - come passo le immagini a predict
        """
        list_predict = []
        path_model = '/content/drive/My Drive/Tesi_Folder/Model_keras_5classes/Model_4.h5'
        model = tf.keras.models.load_model(path_model)
        predict = model.predict(self.list_image, batch_size=50)

        print(predict)


t = time.perf_counter()
sasa = Classification('C:/Users/piero/Test/')
sasa.select_folder()
t1 = time.perf_counter()

s = t1-t
print(s)