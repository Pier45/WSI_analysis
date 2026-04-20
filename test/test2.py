# import openslide
# path = 'D:/Download/map_1.svs'
# slide = openslide.OpenSlide(path)
# image = slide.get_thumbnail((10250, 6250))
# print(image.size)
# image.save('D:/Download/sasa1.png')
#
# a = plt.imread('D:/Download/sasa1.png')
# plt.imshow(a)
# plt.show()

# import os
# os.makedirs('C:/Users/piero/ueue/cap/jjj')
import os
levi = 1
file_path = 'D:/Download/map_4.svs'
start_folder = 'C:/Users/piero/Test/'

def base_folder_manager(levi):
    form = list(file_path).index('.')
    last = len(file_path) - 1 - file_path[::-1].index('/')
    file_name = file_path[last + 1:form]
    folder_name = file_name + '_' + str(levi)
    newpath = start_folder + folder_name
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    path_folder = newpath
    return path_folder


path = base_folder_manager(levi)
print(path)

ddd = ['map_1_2', 'p_0_3_13', 'p_12_15_13', 'p_15_18_13', 'p_18_21_13', 'p_3_6_13', 'p_6_9_13', 'p_9_12_13', 'jjjj']
folders = [y for y in ddd if y[0] == 'p']
print(folders)