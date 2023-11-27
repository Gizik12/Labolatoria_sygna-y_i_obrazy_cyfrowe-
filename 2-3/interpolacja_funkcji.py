import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def h2_kernel(x):
    return np.where((x >= -0.5) & (x < 0.5), 1, 0)

def h3_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

def h4_kernel(x):
    return np.sinc(x/np.pi)

def convolution(x, y, x_interp, kernel):
    y_interp = np.zeros_like(x_interp)

    for i, x_val in enumerate(x_interp):
        for j, x_original in enumerate(x):
            y_interp[i] += y[j] * kernel(x_val - x_original)

    return y_interp / np.sum(kernel(x_interp - x[0]))

def display_results(x_original, y_original, x_interp, y_interp, kernel, mse):
    plt.scatter(x_original, y_original, label='Sinus(x)', s=3)
    plt.scatter(x_interp, y_interp, s=3)
    plt.title(f'Convolution Result with {kernel} Kernel, increased to {len(x_interp)} points')
    plt.legend()

N = 100
x_original = np.linspace(-np.pi, np.pi, N)
y_original = np.sin(x_original)
x_interp = np.linspace(-np.pi, np.pi, N)
y_true = np.sin(x_interp)

result_convolution_h2 = convolution(x_original, y_original, x_interp, h2_kernel)
result_convolution_h3 = convolution(x_original, y_original, x_interp, h3_kernel)
result_convolution_h4 = convolution(x_original, y_original, x_interp, h4_kernel)

mse_h2 = metrics.mean_squared_error(y_true=y_true, y_pred=result_convolution_h2)
mse_h3 = metrics.mean_squared_error(y_true=y_true, y_pred=result_convolution_h3)
mse_h4 = metrics.mean_squared_error(y_true=y_true, y_pred=result_convolution_h4)

plt.figure(figsize=(8, 6))

plt.scatter(x_original, y_original, label='Sinus (x)', s=3)
plt.title('Sinus (x) with {} points'.format(N))
plt.legend()
plt.show()
print("----------------------------------")
display_results(x_original, y_original, x_interp, result_convolution_h2, 'h2(x)', mse_h2)
print("Mse h2 : " + str(mse_h2))
plt.show()
display_results(x_original, y_original, x_interp, result_convolution_h3, 'h3(x)', mse_h3)
print("Mse h3 : "+ str(mse_h3))
plt.show()
display_results(x_original, y_original, x_interp, result_convolution_h4, 'h4(x)', mse_h4)
print("Mse h4 : " + str(mse_h4))
plt.show()
print("----------------------------------")