from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def load_image(addr, i):
    im = Image.open(addr)
    print(im)
    imn = np.asarray(im)
    shape = (128, 128, 3)

    print(imn.shape)
    zaza = plt.imread(addr).shape
    print(zaza[0:2])
    if zaza != shape:
        print('we waglio')
    im.show()


load_image(addr='C:/Users/piero\Test\map_1_2\p_18_21_13/tile_244_18_9.png', i=0)