from PIL import Image
from skimage import data,filters,feature,io,measure
import matplotlib.pyplot as plt
# img = data.camera()
img =io.imread('2.jpg',as_gray=True)

contours = measure.find_contours(img, 0.54)

fig,ax = plt.subplots()
ax.set_axis_off()
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], -contour[:, 0], linewidth=2)

plt.savefig('test.jpg')
# plt.savefig('test.jpg', dpi=None, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype='letter', format=None,
#         transparent=False, pad_inches=0,
#         frameon=None, metadata=None)
# plt.show()