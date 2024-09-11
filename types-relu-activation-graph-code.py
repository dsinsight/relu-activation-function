import numpy as np
import matplotlib.pyplot as plt

# Create input values for plotting
x = np.linspace(-3, 3, 400)

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def rrelu(x, lower=0.01, upper=0.1):
    alpha = np.random.uniform(lower, upper, x.shape)
    return np.where(x > 0, x, alpha * x)

def swish(x, beta=1):
    return x / (1.0 + np.exp(-beta * x))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# List of functions and titles for graphing
activation_functions = [
    (relu, "Standard ReLU"),
    (leaky_relu, "Leaky ReLU"),
    (prelu, "Parametric ReLU (PReLU)"),
    (selu, "Scaled Exponential Linear Unit (SELU)"),
    (elu, "Exponential Linear Unit (ELU)"),
    (rrelu, "Randomized Leaky ReLU (RReLU)"),
    (swish, "Swish"),
    (gelu, "GELU (Gaussian Error Linear Unit)")
]

# Plotting each activation function
for func, title in activation_functions:
    plt.figure()
    plt.plot(x, func(x))
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()
