import numpy as np
import matplotlib.pyplot as plt
import time

N = 10 + 19 # Set N value
k_values = range(N) # Create list of k values

# Compute Fourier coefficient C_k for given k and input signal f
def fourier_coefficient(k, f):
    A_k = (2 / N) * np.sum(f * np.cos(2 * np.pi * k * np.arange(N) / N))
    B_k = (2 / N) * np.sum(f * np.sin(2 * np.pi * k * np.arange(N) / N))
    return A_k + 1j * B_k

# Calculate time and number of operations required to compute Fourier coefficients for all k values
def calculate_time_and_operations(f):
    start_time = time.time()
    operations = 0
    C_k_values = []
    for k in k_values:
        C_k = fourier_coefficient(k, f)
        C_k_values.append(C_k)
        operations += 8 * N + 6
    end_time = time.time()
    calculation_time = end_time - start_time
    return calculation_time, operations

# Plot amplitude and phase spectrum for given input signal f
def plot_spectrum(f):
    C_k_values = [fourier_coefficient(k, f) for k in k_values]
    amplitude_spectrum = np.abs(C_k_values)
    phase_spectrum = np.angle(C_k_values)

    plt.subplot(211)
    plt.plot(amplitude_spectrum)
    plt.title('Amplitude Spectrum')
    plt.subplot(212)
    plt.plot(phase_spectrum)
    plt.title('Phase Spectrum')
    plt.tight_layout()
    plt.show()

f = np.random.rand(N) # Generate random input signal f
calculation_time, operations = calculate_time_and_operations(f)

print('Computing time:', calculation_time)
print('Number of operations:', operations)

plot_spectrum(f)