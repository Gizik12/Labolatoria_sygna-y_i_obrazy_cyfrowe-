import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2

def compress(image, compression_ratio):
    # Dyskretna transformacja Fouriera
    image_fft = fft2(image)
    threshold = np.percentile(np.abs(image_fft), compression_ratio)
    image_fft[np.abs(image_fft) < threshold] = 0
    
    # Odwrotna transformacja Fouriera
    compressed_image = np.real(ifft2(image_fft))
    
    return compressed_image

def main():
    # Wczytanie obrazu w odcieniach szarości
    original_image = plt.imread('image.png')
    # Konwersja do obrazu szaro-skalowego
    gray_image = np.mean(original_image, axis=-1)

    # Wyświetlanie orginalnego obrazu
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Oryginalny obraz')

    # Skompresowany obraz z wybranym współczynnikiem kompresji(%)
    compression_ratio = 90
    compressed_image = compress(gray_image, compression_ratio)

    # Wyświetlanie skompresowany obrazu
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f'Skompresowany obraz ({compression_ratio}% kompresji)')

    plt.show()

main()