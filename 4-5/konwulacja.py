import numpy as np
from skimage import io, color
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# Oryginalny obraz
image = io.imread(r"grzyb.jpg")
print(image.shape)
plt.imshow(image)
plt.title("Orginal Image")
plt.show(block=False)
plt.pause(5)
plt.close()

# Rozmywanie krawędzi przyyciu jądra Laplace
laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

laplace_image = np.dstack([
    convolve(image[:, :, channel], laplace_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

plt.imshow(laplace_image)
plt.title("Laplace Image")
plt.show(block=False)
plt.pause(5)
plt.close()

# Rozmywanie obrazu
blurring_filter = np.full((5, 5), 1/25)

blurring_image = np.dstack([
    convolve(image[:, :, channel], blurring_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

plt.imshow(blurring_image)
plt.title("Blurring Image")
plt.show(block=False)
plt.pause(5)
plt.close()

# Wyostrzanie obrazu
sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

sharpen_image = np.dstack([
    convolve(image[:, :, channel], sharpen_filter, mode="constant", cval=0.0)
    for channel in range(3)
])

plt.imshow(sharpen_image)
plt.title("Sharpen Image")
plt.show(block=False)
plt.pause(5)
plt.close()