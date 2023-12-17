import numpy as np
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plt

image = io.imread(r"Bayer/namib.jpg")

print(image.shape)
demosaicking_convolution_mask = np.dstack([np.ones([2, 2]), 0.5 * np.ones([2, 2]), np.ones([2, 2])]) # R G B

reconstructed_image = np.dstack([
    ndimage.convolve(image[:, :, channel], demosaicking_convolution_mask[:, :, channel], mode="constant", cval=0.0)
    for channel in range(3)
])



plt.imshow(reconstructed_image)
plt.show(block=False)
plt.pause(10)
plt.close()