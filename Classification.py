import os
import PIL
import sys
import time
import threading
import glob
import numpy as np
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt


class Classification:
    def __init__(self, path):
        self.path = path
        self.dictionary = {}
        self.np_list_image = []
        self.shape = (128, 128, 3)
        self.select_folder()

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
        self.dictionary = {}
        list_image = []
        folders_tot = os.listdir(self.path)
        folders = [y for y in folders_tot if y[0] == 'p']
        print(folders)
        for i in folders:
            sel_folder = self.path + str(i) + '/' + '*.png'
            list_files, n_elements = self.analysis_folder(sel_folder)
            for j in list_files:
                sub_d = {}
                complete_name = j[list(j).index('\\') + 1:-4]
                partial = complete_name[list(complete_name).index('_') + 1:]
                n_tile = int(partial[:partial.index('_')])
                tile_pos = partial[list(partial).index('_') + 1:]
                column = int(tile_pos[:list(tile_pos).index('_')])
                row = int(tile_pos[list(tile_pos).index('_')+1:])
                np_image, shape_x, shape_y = self.tile_control(j)
                sub_d['im_path'], sub_d['shape_x'], sub_d['shape_y'], sub_d['col'], sub_d['row'] = j, shape_x, shape_y, column, row
                list_image.append(np_image)
                self.dictionary[n_tile] = sub_d
            print('Selected Folder:   {:<40} Number of elements: {}'.format(sel_folder, n_elements))

        self.np_list_image = np.asarray(list_image)
        print(self.np_list_image.shape)

    def tile_control(self, j):
        """ This method read the image, modify it as numpy array and at the end control if same padding is needed."""
        image = misc.imread(j)
        np_image = np.asarray(image/255, dtype=float)
        shape_i = image.shape
        if shape_i != self.shape:
            try:
                np_image = np.pad(np_image, ((0, self.shape[0] - shape_i[0]), (0, self.shape[1] - shape_i[1]), (0, 0)),
                                  'constant', constant_values=1)
            except ValueError:
                print('ATTENTION: error in the png file shape.\n Shape tile standard:{} '
                      '\n Shape tile:{}'.format(self.shape, shape_i))
        return np_image, shape_i[0], shape_i[1]

    def classify(self):
        """
        Load the model and analyze the tile, the dictionary is updated with the predicted label
        """

        path_model = 'D:/Download/Model_1_512.h5'
        model = tf.keras.models.load_model(path_model)
        np_image = np.asarray(self.np_list_image)
        print(np_image.shape)
        predict = model.predict(np_image, batch_size=50)
        probs = np.asarray(predict)
        clas = np.argmax(probs, axis=1)
        print(clas.shape)

        for i, y in enumerate(self.dictionary):
            self.dictionary[y]['class'] = clas[i]

        print(self.dictionary)
        somma = np.sum(probs, axis=0)
        print(clas[0])

    def show_image(self, im):
        """GREAT NOTE: LIST ARE index-1 for the 0 index """
        plt.subplot(1, 3, 1)
        plt.imshow(self.np_list_image[76, :, :, :])
        plt.subplot(1, 3, 2)
        #plt.imshow(self.np_list_image[75, :, :, :])
        plt.imshow(im)
        plt.subplot(1, 3, 3)
        plt.imshow(self.np_list_image[74, :, :, :])
        plt.show()

    def overlay(self):
        a = plt.imread(self.path + '/thumbnail/th.png')
        sel_res = a.shape
        image_base = np.zeros((sel_res[0], sel_res[1], 4), dtype=float)
        print(f'IMAGE SHAPE BASE {image_base.shape}')
        step = 32

        n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0

        for i, name_t in enumerate(self.dictionary):

            shape_x = self.dictionary[name_t]['shape_x']
            shape_y = self.dictionary[name_t]['shape_y']
            column = self.dictionary[name_t]['col']
            row = self.dictionary[name_t]['row']
            clas = self.dictionary[name_t]['class']
            print('+++ shapex: {} shapey: {} +++'.format(shape_x, shape_y))
            print(column, row, step)
            c0 = column*step
            r0 = row*step
            print(' R0: {:>5d}  RFIN: {:>5d} \n C0:{:>5d}   CFIN: {:>5d}'.format(r0, r0+shape_y, c0, c0+shape_x))

            if clas == 1:
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.3
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += 0.5
                n1 += 1
            elif clas == 2:
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.3
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += 0.5
                n2 += 1
            elif clas == 3:
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += 0.5
                n3 += 1
            elif clas == 4:
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.5
                n4 += 1
            elif clas == 5:
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += 0.5
                n5 += 1

            # trasparence layer
            image_base[r0:r0 + shape_x, c0:c0 + shape_y, 3] = 0.7

        print('N1 --> {:>4}\nN2 --> {:>4}\nN3 --> {:>4}\nN4 --> {:>4}\nN5 --> {:>4}'.format(n1, n2, n3, n4, n5))
        print(n1 + n2 + n3 + n4 + n5)
        plt.imshow(a)
        plt.imshow(image_base.astype(np.float))
        plt.axis('off')
        plt.show()
        #plt.savefig('D:/Download/sasa.png', bbox_inches='tight', pad_inches=0)


t = time.perf_counter()
sasa = Classification('C:/Users/piero/Test/map_1_1/')
#sasa.show_image()
sasa.classify()
sasa.overlay()
t1 = time.perf_counter()

s = t1-t
print(s)