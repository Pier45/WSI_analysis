import os
import time
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import json
from PIL import Image


class Classification:
    def __init__(self, path, ty):
        self.path = path
        self.ty = ty
        self.dictionary = {}
        self.np_list_image = []
        self.shape = (64, 64, 3)
        self.select_folder()
        self.cl = ['AC', 'H', 'AD']

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
        n_tile = 0
        list_image = []
        folders_tot = os.listdir(self.path)
        if self.ty == 'analysis':
            folders = [y for y in folders_tot if y[0:2] == 'p_']
        else:
            folders = ['/AC', '/H', '/AD']

        for i in folders:
            sel_folder = self.path + str(i) + '/*.png'
            list_files, n_elements = self.analysis_folder(sel_folder)
            for j in list_files:
                sub_d = {}
                complete_name = j[j.index('tile'):-4]
                partial = complete_name[complete_name.index('_') + 1:]
                tile_pos = partial[list(partial).index('_') + 1:]
                column = int(tile_pos[:list(tile_pos).index('_')])
                row = int(tile_pos[list(tile_pos).index('_')+1:])
                np_image, shape_x, shape_y = self.tile_control(j)
                if self.ty == 'datacleaning':
                    sub_d['name'] = j[j.index('pz_'):j.index('_tile')]
                    sub_d['true_class'] = i[1:]
                    n_tile += 1
                else:
                    n_tile = partial[:partial.index('_')]
                sub_d["im_path"], sub_d["shape_x"], sub_d["shape_y"], sub_d["col"], sub_d["row"] = j, shape_x, shape_y, column, row
                list_image.append(np_image)
                self.dictionary[n_tile] = sub_d
            print('Selected Folder:   {:<40} Number of elements: {}   elementi{}'.format(sel_folder, n_elements, len(list_image)))

        print(len(self.dictionary))
        self.np_list_image = np.asarray(list_image)

    def tile_control(self, j):
        """ This method read the image, modify it as numpy array and at the end control if same padding is needed."""
        np_image = imread(j)
        shape_i = np_image.shape
        if shape_i != self.shape:
            try:
                np_image = np.pad(np_image, ((0, self.shape[0] - shape_i[0]), (0, self.shape[1] - shape_i[1]), (0, 0)),
                                  'constant', constant_values=1)
            except ValueError:
                print('ATTENTION: error in the png file shape.\n Shape tile standard:{} '
                      '\n Shape tile:{}'.format(self.shape, shape_i))
        return np_image, shape_i[0], shape_i[1]

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def classify(self, typean, monte_c, model_n, progress_callback, view=0):
        """
        Load the model and analyze the tile, the dictionary is updated with the predicted label
        """
        self.load_model(model_name=model_n)
        print(self.np_list_image.shape)
        tesu = []
        progress_callback.emit(1)
        mc = monte_c

        for i in range(0, mc):
            if i != 0:
                progress_callback.emit(100*i/6)

            tesu.append(self.model.predict(self.np_list_image, batch_size=50))

        probs = np.asarray(tesu)
        clas_mean = np.mean(probs, axis=0)
        aleatoric = np.mean(probs * (1 - probs), axis=0)
        epistemic = np.mean(probs ** 2, axis=0) - np.mean(probs, axis=0) ** 2
        print('TESU: {} \n SHAPE EPI: {} \n SHAPE ALE: {}'.format(len(tesu), epistemic.shape, aleatoric.shape))

        for i, y in enumerate(self.dictionary):
            self.dictionary[y]["pred_class"] = self.cl[int(np.argmax(clas_mean[i]))]
            self.dictionary[y]["epi"] = float(np.round(np.sum(epistemic[i]), 4))
            self.dictionary[y]["ale"] = float(np.round(np.sum(aleatoric[i]), 4))

        if typean == 'datacleaning':
            progress_callback.emit(100)
        else:
            self.overlay(typean)
            progress_callback.emit(80)
            self.overlay(typean, unc='epi')
            progress_callback.emit(90)
            self.overlay(typean, unc='ale')
            self.overlay(typean, unc='tot')
            progress_callback.emit(100)

        print('dictionary', len(self.dictionary))
        with open(os.path.join(self.path, 'dictionary_monte_' + str(monte_c) + '_js.txt'), 'w') as f:
            json.dump(self.dictionary, f, indent=4)

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
        image_base_H = np.zeros((a.shape[0], a.shape[1], 4), dtype=float)
        image_base_AC = np.zeros((a.shape[0], a.shape[1], 4), dtype=float)
        image_base_AD = np.zeros((a.shape[0], a.shape[1], 4), dtype=float)

        print(f'IMAGE SHAPE BASE {image_base.shape}')
        step = 64    # per casi di rimpicciolimero grandezza tiles diviso quando si vuole es 128 / 4 = 32
        res_path = self.path + '/result'

        if not os.path.exists(res_path):
            os.makedirs(res_path + '/uncertainty')

        if unc == 'Pred_class':
            res_name = [self.path+'result/'+str(unc)+'.png', self.path+'result/AC.png', self.path+'result/H.png', self.path+'result/AD.png']
        else:
            res_name = self.path + 'result/uncertainty/' + str(unc) + '.png'

        n1, n2, n3 = 0, 0, 0

        for i, name_t in enumerate(self.dictionary):

            shape_x = int(self.dictionary[name_t]["shape_x"]) #/4
            shape_y = int(self.dictionary[name_t]["shape_y"]) #/4
            column = self.dictionary[name_t]["col"]
            row = self.dictionary[name_t]["row"]
            c0 = column*step
            r0 = row*step
            if unc == 'Pred_class':
                clas = self.dictionary[name_t]["pred_class"]
                if clas == 'AC':
                    # red
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += 1
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += 0.2
                    image_base_AC[r0:r0 + shape_x, c0:c0 + shape_y, 0] += 1
                    image_base_AC[r0:r0 + shape_x, c0:c0 + shape_y, 2] += 0.2

                    n1 += 1
                elif clas == 'H':
                    # green
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.95
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 0] += 0.23
                    image_base_H[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.95
                    image_base_H[r0:r0 + shape_x, c0:c0 + shape_y, 0] += 0.23
                    n2 += 1
                elif clas == 'AD':
                    # blue
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += 0.9
                    image_base[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.5
                    image_base_AD[r0:r0 + shape_x, c0:c0 + shape_y, 2] += 0.9
                    image_base_AD[r0:r0 + shape_x, c0:c0 + shape_y, 1] += 0.5
                    n3 += 1

            elif unc == 'epi':
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += abs(self.dictionary[name_t]["epi"])
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 3] += abs(self.dictionary[name_t]["epi"])
            elif unc == 'ale':
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += abs(self.dictionary[name_t]["ale"])
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 3] += abs(self.dictionary[name_t]["ale"])
            elif unc == 'tot':
                u_tot = self.dictionary[name_t]["ale"]+self.dictionary[name_t]["epi"]
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 2] += abs(u_tot)
                image_base[r0:r0 + shape_x, c0:c0 + shape_y, 3] += abs(u_tot)
            else:
                print(f'Strange command:{unc}')
                pass

        if unc == 'Pred_class':
            print('AC --> {:>4}\nH --> {:>4}\nAD --> {:>4}'.format(n1, n2, n3))
            print(n1 + n2 + n3)
            image_base[:, :, 3] = 0.3

            image_base_AC[:, :, 3] = 0.4
            rc, cc = np.where(image_base_AC[:, :, 0] < 0.2)
            image_base_AC[rc, cc, 3] = 0.1

            image_base_AD[:, :, 3] = 0.3
            rd, cd = np.where(image_base_AD[:, :, 2] < 0.2)
            image_base_AD[rd, cd, 3] = 0.1

            image_base_H[:, :, 3] = 0.3
            rh, ch = np.where(image_base_H[:, :, 1] < 0.2)
            image_base_H[rh, ch, 3] = 0.1

            list_im = [image_base, image_base_AC, image_base_H, image_base_AD]

            for im, name in enumerate(res_name, 0):
                self.new_save(list_im[im], name)
        else:
            if unc == 'tot' or unc == 'ale':
                r_h, c_h = np.where(image_base[:, :, 2] < 0.4)
                image_base[r_h, c_h, 2] = 0
                image_base[r_h, c_h, 3] = 0.1
                r_m, c_m = np.where((image_base[:, :, 2] < 0.5) & (image_base[:, :, 2] > 0.4))
                image_base[r_m, c_m, 3] = 0.2
            else:
                pass

            self.new_save(image_base, res_name)

    def new_save(self, image_base, res_name):
        image_base = np.where(image_base < 1, image_base, 1)
        background = Image.open(self.path + '/thumbnail/th.png')
        foreground = Image.fromarray(np.uint8(image_base*255), mode='RGBA')
        background.paste(foreground, (0, 0), foreground)
        #background.show()
        background.save(res_name)

    def load_dict(self):
        name_f = os.path.join(self.path, 'dictionary_monte_5_js.txt')
        with open(name_f, 'r') as f:
            self.dictionary = json.load(f)


if __name__ == '__main__':

    t = time.perf_counter()
    sasa = Classification('C:/Users/piero/Test/31400_2/', ty='analysis')
    #sasa.show_image()
    #sasa.classify(typean='fast')
    sasa.load_dict()
    sasa.overlay(unc='epi')
    t1 = time.perf_counter()

    s = t1-t
    print(s)