import numpy as np
import time

N = 10 + 19 # Set N value
k_values = range(N) # Create list of k values

# Compute Fourier coefficient C_k for given k and input signal f
def fourier_coefficient(k, f):
    A_k = (2 / N) * np.sum(f * np.cos(2 * np.pi * k * np.arange(N) / N))
    B_k = (2 / N) * np.sum(f * np.sin(2 * np.pi * k * np.arange(N) / N))
    return A_k + 1j * B_k

# Compute Fourier coefficients for given input signal f by Cooley–Tukey FFT
def fft(x):
    n = len(x)
    if n == 1:
        return x
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi / n)
        return np.concatenate([even + factor * odd, even - factor * odd])

# Calculate time and number of operations required to compute Fourier coefficients for all k values by DFT
def dft_calculate_time_and_operations(f):
    start_time = time.time()
    add_operations = 0
    mult_operations = 0
    C_k_values = []
    for k in k_values:
        C_k = fourier_coefficient(k, f)
        C_k_values.append(C_k)
        add_operations += 1 + 2 * N
        mult_operations += 1 + 2 * (3 + 2 * N)
    end_time = time.time()
    calculation_time = end_time - start_time
    return calculation_time, add_operations, mult_operations

# Calculate time and number of operations required to compute Fourier coefficients for all k values by FFT
def fft_calculate_time_and_operations(f):
    start_time = time.time()
    fft(f)
    end_time = time.time()
    add_operations = N * np.log2(N)
    mult_operations = (N + 1) * np.log2(N)
    calculation_time = end_time - start_time
    return calculation_time, int(add_operations), int(mult_operations)

f = np.random.rand(N) # Generate random input signal f

# Print the results of DFT
calculation_time, add_ops, mult_ops = dft_calculate_time_and_operations(f)
print('Results of DFT.')
print('Computing time:', calculation_time)
print(f'Number of  addition operations: {add_ops}')
print(f'Number of  multiplication operations: {mult_ops}')

# Print the results of FFT
calculation_time, add_ops, mult_ops = fft_calculate_time_and_operations(f)
print('\nResults of FFT.')
print('Computing time:', calculation_time)
print(f'Number of  addition operations: {add_ops}')
print(f'Number of  multiplication operations: {mult_ops}')