import numpy as np
import matplotlib.pyplot as plt
import json

# Define the function to be approximated
def f(x):
    return 19 * np.sin(np.pi * 19 * x)

# Define the range of x values to use
x = np.linspace(-np.pi, np.pi, 1000)

# Plot the function f(x)
def plot_f(x):
    y = f(x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y = 19 * sin(pi * 19 * x)")
    plt.show()

# Call the plot_f function to plot f(x)
plot_f(x)

# Compute the Fourier coefficients of f(x) using numerical integration
def fourier_coefficients(x, N):
    a0 = (2 / np.pi) * np.trapz(f(x), x)
    ak = []
    bk = []
    for k in range(0, N):
        ak.append((2 / np.pi) * np.trapz(f(x) * np.cos(k * x), x))
        bk.append((2 / np.pi) * np.trapz(f(x) * np.sin(k * x), x))
    return a0, ak, bk

# Define the number of terms
N = 50
# Compute and store the first 50 Fourier coefficients of f(x)
a0, ak, bk = fourier_coefficients(x, N)

# Print the Fourier coefficients in a table
def print_coefficients_table(a0, ak, bk):
    print("{:^8s} {:^15s} {:^15s}".format("k", "ak", "bk"))
    print("{:^8s} {:^15.10f} {:^15s}".format("0", a0, "-"))
    for k in range(0, len(ak)):
        print("{:^8d} {:^15.10f} {:^15.10f}".format(k + 1, ak[k], bk[k]))

# Call the print_coefficients_table function to print the Fourier coefficients
print_coefficients_table(a0, ak ,bk)

# Compute the Fourier series approximation of f(x) using N terms
def fourier_series_approximation(x, N):
    series_sum = a0 / 2
    for k in range(1, N + 1):
        series_sum += ak[k - 1] * np.cos(k * x) + bk[k - 1] * np.sin(k * x)
    return series_sum

# Call the print_coefficients_table function to calculate approximations
approximations = fourier_series_approximation(x, N)

# Plot the original function and its Fourier series approximation using N terms
def plot_approximation(x, N):
    y = f(x)
    s = np.zeros_like(x)
    plt.figure(figsize=(15, 10))
    plt.plot(x, y, label='Original function')
    for i in range(0, N, 4):
        s += ak[i] * np.cos(i * x) + bk[i] * np.sin(i * x)
        plt.plot(x, s, label=f'N={i + 1}')
    plt.xlabel("x")
    plt.ylabel("Fourier approximation")
    plt.legend()
    plt.show()

# Call the plot_approximation function to plot the original function and its Fourier series approximation using N terms
plot_approximation(x, N)

# Plot the first N harmonics and their corresponding Fourier coefficients
def plot_harmonics(a0, ak, bk, N):
    k_values = list(range(N + 1))
    a_values = [a0] + ak
    b_values = [0] + bk
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.stem(k_values, a_values, linefmt='r-', markerfmt='ro', label='a(k)')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('a(k) in the frequency domain')
    plt.subplot(122)
    plt.stem(k_values, b_values, linefmt='b-', markerfmt='bo', label='b(k)')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('b(k) in the frequency domain')
    plt.show()

# Call the plot_harmonics function to plot the first N harmonics and their corresponding Fourier coefficients
plot_harmonics(a0, ak,bk, N)

# Compute the relative error of the Fourier series approximation using N terms
def relative_errors(x, approximations):
    y = f(x)
    errors = np.abs(y - approximations) / np.abs(y)
    return errors

# Call the relative_errors function to calculate relative error for each value of f(x)
errors = relative_errors(x, approximations)
# Compute the mean of relative errors
error = np.mean(errors)

# Plot the relative error of the Fourier series approximation using N terms
def plot_relative_error(x, errors):
    plt.plot(x, errors)
    plt.xlabel('x')
    plt.ylabel('Relative error')
    plt.show()

# Call the plot_relative_error function to plot the relative error of the Fourier series approximation using N terms
plot_relative_error(x, errors)

# Save the order N, the Fourier coefficients, and the relative error to a JSON file
def save_to_json(N, a0, ak, bk, error, filename):
    data = {
        'N': N,
        'a0': a0,
        'ak': ak,
        'bk': bk,
        'error': error
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


# Call the save_to_json function to save the results to a JSON file
save_to_json(N, a0, ak, bk, error, 'results.json')
