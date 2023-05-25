import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

n = 19; N = 100 * n  # Set N to desired value
A = 1  # Set amplitude
phi = 1  # Set phase shift

# Generate test sequence with distorted sine wave
def generate_test_sequence(N, n, A, phi):
    x = np.linspace(0, 0.555, N)
    exact = A * np.sin(n * x + phi)
    deviation = np.random.uniform(-0.05 * A, 0.05 * A, N)
    apprx = exact + deviation
    return x, exact, apprx

# Compute the average values
def arithmetic_mean(values):
    return np.mean(values)

def harmonic_mean(values):
    values = np.where(values == 0, np.nan, values)
    return len(values) / np.nansum(1.0 / values)

def geometric_mean(values):
    values = np.where(values <= 0, np.nan, values)
    return np.nanprod(values) ** (1.0 / len(~np.isnan(values)))

# Calculate absolute and relative errors
def calculate_errors(exact, apprx):
    absolute_error = np.abs(exact - apprx)
    relative_error = absolute_error / (np.abs(exact) + 1)
    return absolute_error, relative_error

# Plot both the exact and approximate sequences
def plot_sequences(x, exact, apprx):
    # Plot sequences separately
    fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')
    ax1.plot(x, exact, color='b')
    ax1.set_title('Exact', loc='left')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax2.plot(x, apprx, color='m')
    ax2.set_title('Approximate', loc='left')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    plt.show()
    # Plot overlapping sequences
    plt.plot(x, apprx, label='Approximate', color='m')
    plt.plot(x, exact, label='Exact', color='b')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# Generate sequences
x, exact, apprx = generate_test_sequence(N, n, A, phi)

# Plot sequences
plot_sequences(x, exact, apprx)

# Calculate the mean values
mean_table = PrettyTable()
mean_table.field_names = ["Mean", "Exact", "Approximate"]
mean_table.add_rows([
    ["Arithmetic", arithmetic_mean(exact), arithmetic_mean(apprx)],
    ["Harmonic", harmonic_mean(exact), harmonic_mean(apprx)],
    ["Geometric", geometric_mean(exact), geometric_mean(apprx)]
])
print(mean_table)

# Calculate errors
absolute_error, relative_error = calculate_errors(exact, apprx)

# Compare errors
error_table = PrettyTable()
error_table.field_names = ["Error", "Minimum", "Maximum"]
error_table.add_rows([
    ['Absolute', np.min(absolute_error), np.max(absolute_error)],
    ['Relative', np.min(relative_error), np.max(relative_error)]
])
print(error_table)
