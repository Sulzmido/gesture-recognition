import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale

dog = imread('Images/dog1.jpg', as_gray=True)
print(dog.shape)
# https://en.wikipedia.org/wiki/German_Shepherd#/media/File:Kim_at_14_weeks.jpg

# scale down the image to one third
dog = rescale(dog, 1 / 3, mode='reflect')
# calculate the hog and return a visual representation.
dog_hog, dog_hog_img = hog(
    dog, pixels_per_cell=(12, 12),
    cells_per_block=(2, 2),
    orientations=8,
    visualize=True,
    block_norm='L2-Hys')

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(8, 6)
# remove ticks and their labels
[a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
 for a in ax]

ax[0].imshow(dog, cmap='gray')
ax[0].set_title('dog')
ax[1].imshow(dog_hog_img, cmap='gray')
ax[1].set_title('hog')
plt.show()