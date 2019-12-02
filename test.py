import numpy as np
np.set_printoptions(precision=2)
aa = np.random.random((5,2))
print(aa)

print(aa[0:3,:])
print(aa[3:5,:])

import os
cwd = os.getcwd()
print(cwd)

from scipy import misc
import glob


image = misc.imread('C:/Users/piero/Test/p_9_12_14\\tile_118_9_0.png')
print(image.shape)
print(image.dtype)


a = 'C:/Users/piero/Test/p_9_12_14\\tile_118_9_0.png'
complete_name = a[list(a).index('\\')+1:-4]
print(complete_name)
partial = complete_name[list(complete_name).index('_') + 1:]
print(partial)
n_tile = partial[:partial.index('_')]
print(n_tile)

