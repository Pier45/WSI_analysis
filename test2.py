import openslide
path = 'D:/Download/map_1.svs'
slide = openslide.OpenSlide(path)
image = slide.get_thumbnail((10250, 6250))
print(image.size)
image.save('D:/Download/sasa1.png')

a = plt.imread('D:/Download/sasa1.png')
plt.imshow(a)
plt.show()