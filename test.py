import numpy as np
shape_i = (5, 5, 3)

image = np.zeros((5, 5, 3))
shape = image.shape

print(shape_i, shape)
np_image = np.pad(image, ((0, shape_i[0] - shape[0]), (0, shape_i[1] - shape[1]), (0, 0)), 'constant', constant_values=1)
print(np_image.shape)
print(np_image)

if shape_i != shape:
    print('sdada')
else:
    print('ok')