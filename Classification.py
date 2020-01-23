import os
import time
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
        self.shape = (64, 64, 3)
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
        the image vectors already padded. Only the folder that start with 'p_' are analayzed, others are skipped.
        """
        self.dictionary = {}
        list_image = []
        folders_tot = os.listdir(self.path)
        folders = [y for y in folders_tot if y[0:2] == 'p_']
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

    def load_model(self, state=9):
        path_model = 'Model_1_85aug.h5'
        self.model = tf.keras.models.load_model(path_model)

        if state == 0:
            self.classify()
            self.overlay()
            self.overlay(unc='epi')
            self.overlay(unc='ale')
            self.overlay(unc='tot')

    def classify(self):
        """
        Load the model and analyze the tile, the dictionary is updated with the predicted label
        """

        np_image = np.asarray(self.np_list_image)
        print(np_image.shape)
        tesu = []
        for i in range(0, 5):
            print(i)
            tesu.append(self.model.predict(np_image, batch_size=50))

        probs = np.asarray(tesu)
        clas_mean = np.mean(probs, axis=0)
        aleatoric = np.mean(probs * (1 - probs), axis=0)
        epistemic = np.mean(probs ** 2, axis=0) - np.mean(probs, axis=0) ** 2
        print('SHAPE EPI: {} \n SHAPE ALE: {}'.format(epistemic.shape, aleatoric.shape))

        for i, y in enumerate(self.dictionary):
            self.dictionary[y]['class'] = np.argmax(clas_mean[i])
            self.dictionary[y]['epi'] = np.sum(epistemic[i])
            self.dictionary[y]['ale'] = np.sum(aleatoric[i])

        print(self.dictionary)

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

    def overlay(self, unc='Pred_class'):
        a = plt.imread(self.path + '/thumbnail/th.png')
        image_base = np.zeros((a.shape[0], a.shape[1], 4), dtype=float)
        print(f'IMAGE SHAPE BASE {image_base.shape}')
        step = 64    # per casi di rimpicciolimero grandezza tiles diviso quando si vuole es 128 / 4 = 32
        res_path = self.path + '/result'

        if not os.path.exists(res_path):
            os.makedirs(res_path + '/uncertainty')

        if unc == 'Pred_class':
            res_name = self.path + 'result/' + str(unc) + '.png'
        else:
            res_name = self.path + 'result/uncertainty/' + str(unc) + '.png'

        n1, n2, n3 = 0, 0, 0

        for i, name_t in enumerate(self.dictionary):

            shape_x = int(self.dictionary[name_t]['shape_x']) #/4
            shape_y = int(self.dictionary[name_t]['shape_y']) #/4
            column = self.dictionary[name_t]['col']
            row = self.dictionary[name_t]['row']
            c0 = column*step
            r0 = row*step
            if unc == 'Pred_class':
                clas = self.dictionary[name_t]['class']
                if clas == 0:
                    # red
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += 0.5
                    n1 += 1
                elif clas == 1:
                    # green
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.5
                    n2 += 1
                elif clas == 2:
                    # blue
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += 0.5
                    n3 += 1

            elif unc == 'epi':
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += abs(self.dictionary[name_t]['epi'])
            elif unc == 'ale':
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += abs(self.dictionary[name_t]['ale'])
            elif unc == 'tot':
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += abs(self.dictionary[name_t]['ale'])
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += abs(self.dictionary[name_t]['epi'])
            else:
                print(f'Strange command:{unc}')
                pass

        if unc == 'Pred_class':
            print('AC --> {:>4}\nH --> {:>4}\nAD --> {:>4}'.format(n1, n2, n3))
            print(n1 + n2 + n3)
            image_base[:, :, 3] = 0.4
        else:
            image_base[:, :, 3] = 0.7

        image_base = np.where(image_base < 1, image_base, 1)

        my_dpi = 200
        plt.figure(figsize=(a.shape[0] / my_dpi, a.shape[1] / my_dpi), dpi=my_dpi, frameon=False)
        plt.imshow(a)
        plt.imshow(image_base.astype(np.float))
        plt.axis('off')

        plt.savefig(res_name, dpi=my_dpi, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':

    t = time.perf_counter()
    sasa = Classification('C:/Users/piero/Test/31400_2/')
    #sasa.show_image()
    sasa.load_model()
    sasa.classify()
    sasa.overlay(unc='Pred_class')
    t1 = time.perf_counter()

    s = t1-t
    print(s)