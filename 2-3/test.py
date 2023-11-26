import numpy as np
import matplotlib.pyplot as plt

# Funkcje do interpolacji
def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(x - 1)

def f3(x):
    return np.sign(np.sin(8 * x))

# Funkcje jądrowe
def h1(x):
    return np.where((x >= 0) & (x <= 1), 1, 0)

def h2(x):
    return np.where((x >= -1/2) & (x <= 1/2), 1, 0)

def h3(x):
    return np.where((x >= -1) & (x <= 1), 1 - np.abs(x), 0)

# Interpolacja funkcji za pomocą konwolucji
def interpolate(f, h, N):
    x = np.linspace(-np.pi, np.pi, N)
    y = f(x)

    kernel = h(np.linspace(0, 1, N))

    interpolated_y = np.convolve(y, kernel, mode='same') / np.sum(kernel)

    return x, interpolated_y

# Kryterium MSE
def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

# Wykonanie interpolacji i obliczenie MSE dla różnych funkcji i jąder
functions = [f1, f2, f3]
kernels = [h1, h2, h3]
N_values = [100, 200, 400, 1000]

for f in functions:
    for h in kernels:
        print(f"Interpolacja funkcji {f.__name__} z jądrem {h.__name__}:")
        for N in N_values:
            x, interpolated_y = interpolate(f, h, N)
            mse_value = mse(f(x), interpolated_y)
            print(f"N = {N}, MSE = {mse_value}")

            # Wykresy
            plt.figure()
            plt.plot(x, f(x), label='Oryginał')
            plt.plot(x, interpolated_y, label='Interpolacja')
            plt.legend()
            plt.title(f"{f.__name__} - {h.__name__} - N={N}")
            plt.show()