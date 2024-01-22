import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2

def compress(image, compression_ratio):
    # Wykonaj dyskretną transformację Fouriera
    image_fft = fft2(image)
    
    # Ustal granicę odcięcia na podstawie współczynnika kompresji
    threshold = np.percentile(np.abs(image_fft), compression_ratio)
    
    # Wyzeruj współczynniki poniżej progu
    image_fft[np.abs(image_fft) < threshold] = 0
    
    # Wykonaj odwrotną transformację Fouriera
    compressed_image = np.real(ifft2(image_fft))
    
    return compressed_image

def main():
    # Wczytaj obraz w odcieniach szarości (przykład)
    original_image = plt.imread('image.png')
    gray_image = np.mean(original_image, axis=-1)  # Konwersja do obrazu szaro-skalowego

    # Wyświetl oryginalny obraz
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Oryginalny obraz')

    # Skompresuj obraz z wybranym współczynnikiem kompresji (np. 95%)
    compression_ratio = 95
    compressed_image = compress(gray_image, compression_ratio)

    # Wyświetl skompresowany obraz
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f'Skompresowany obraz ({compression_ratio}% kompresji)')

    plt.show()

main()