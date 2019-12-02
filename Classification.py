import os
import PIL
import sys
import multiprocessing
import time
from math import ceil
import threading
from random import shuffle
import glob
import numpy as np
from scipy import misc


class Classification:
    def __init__(self, path):
        self.path = path

    def analysis_folder(self, sel_folder):
        list_files = glob.glob(sel_folder)
        n_elements = len(list_files)
        return list_files, n_elements

    def select_folder(self):
        dictionary, sub_dic = {}, {}
        folders = os.listdir(self.path)
        print(folders)
        for i in folders:
            sel_folder = self.path + str(i) + '/' + '*.png'
            list_files, n_elements = self.analysis_folder(sel_folder)
            for j in list_files:
                complete_name = j[list(j).index('\\') + 1:-4]
                partial = complete_name[list(complete_name).index('_') + 1:]
                n_tile = partial[:partial.index('_')]
                image = misc.imread(j)
                np_image = np.asarray(image/255, dtype=float)
                sub_dic['image'] = np_image
                dictionary[n_tile] = sub_dic
            print('Selected Folder:   {:<40} Number of elements: {}'.format(sel_folder, n_elements))
        print(dictionary['77'])

    def classify(self):
        """Load the model and analyze the tile, the dictionary is updated with the predicted label"""
        pass


t = time.perf_counter()
sasa = Classification('C:/Users/piero/Test/')
sasa.select_folder()
t1 = time.perf_counter()

s = t1-t
print(s)