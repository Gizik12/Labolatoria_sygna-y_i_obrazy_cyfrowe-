import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

image = io.imread("sebastian-kullmann-300x300.jpg")
gray_image = color.rgb2gray(image)


def resize_image(image, kernel, stride):
    height_kernel, width_kernel = kernel.shape
    height_image, width_image = image.shape
    height_result = height_image // stride
    width_result = width_image // stride
    result = np.zeros((height_result, width_result))

    for i in range(0, height_image - height_kernel + stride, stride):
        for j in range(0, width_image - width_kernel + stride, stride):
            result[i // stride, j // stride] = np.sum(image[i:i + height_kernel, j:j + width_kernel] * kernel)

    return result


def enlarge_image(image, factor):
    height, width = image.shape
    enlarged_height = height * factor
    enlarged_width = width * factor
    enlarged_image = np.zeros((enlarged_height, enlarged_width))

    for i in range(0, enlarged_height):
        for j in range(0, enlarged_width):
            original_i = i / factor
            original_j = j / factor

            floor_i = int(original_i)
            ceil_i = min(floor_i + 1, height - 1)
            floor_j = int(original_j)
            ceil_j = min(floor_j + 1, width - 1)

            weight_i = original_i - floor_i
            weight_j = original_j - floor_j

            pixel_value = (1 - weight_i) * (1 - weight_j) * image[floor_i, floor_j] + \
                          weight_i * (1 - weight_j) * image[ceil_i, floor_j] + \
                          (1 - weight_i) * weight_j * image[floor_i, ceil_j] + \
                          weight_i * weight_j * image[ceil_i, ceil_j]

            enlarged_image[i, j] = pixel_value

    return enlarged_image


plt.figure(figsize=(10, 6))
plt.imshow(gray_image, cmap='gray')
plt.title("Original Grayscale Image")
plt.show()

kernel1 = np.full((2, 2), 1 / 4)
resized_image1 = resize_image(gray_image, kernel1, 2)

plt.figure(figsize=(10, 6))
plt.imshow(resized_image1, cmap='gray')
plt.title("Resized Grayscale Image (Kernel Size: 2x2)")
plt.show()

kernel2 = np.full((3, 3), 1 / 9)
resized_image2 = resize_image(gray_image, kernel2, 3)

plt.figure(figsize=(10, 6))
plt.imshow(resized_image2, cmap='gray')
plt.title("Resized Grayscale Image (Kernel Size: 3x3)")
plt.show()

enlarged_image = enlarge_image(resized_image2, 3)

plt.figure(figsize=(10, 6))
plt.imshow(enlarged_image, cmap='gray')
plt.title("Enlarged Grayscale Image")
plt.show()
