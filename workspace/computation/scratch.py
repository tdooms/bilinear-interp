# %%

import os 
print(os.getcwd())

# %%

import torch
from einops import *
import plotly.express as px

e = torch.randn(5, 3)
v = torch.randn(3, 7)
w = torch.randn(3, 7)

e_v = einsum(e, v, "i h, h o -> i o")
e_w = einsum(e, w, "i h, h o -> i o")
e_wv = einsum(e_w, e_v, "i1 o, i2 o -> o i1 i2")

# wv = einsum(w, v, "h1 o, h2 o -> o h1 h2")
# e_wv2 = einsum(e, wv, "i h, o h1 h2 -> o i1 i2")

# %%
import math
import torch

k = 5

angles = (2*math.pi / k) * torch.arange(k)
e = torch.stack([angles.cos(), angles.sin()])
norm = 2/k
err = (e.T @ e)*norm - torch.eye(e.shape[1])
print(f"Reconstruction MSE for regular polygon: {(err**2).mean()}")


angles = (2*math.pi) * torch.rand(k)
e = torch.stack([angles.cos(), angles.sin()])
einv = torch.linalg.pinv(e)
err = einv @ e - torch.eye(e.shape[1])
print(f"Reconstruction MSE for random angles: {(err**2).mean()}")

# %%
# I'm too dumb to derive closed formula for this

import numpy as np
from scipy.optimize import minimize_scalar

def objective_function(a):
    return np.sum((x/2 - a*x**2)**2)

x = np.linspace(0, 1, 100)

result = minimize_scalar(objective_function).x
print("Optimal value of a:", result)

# %%
from scipy.optimize import minimize
import numpy as np

# Define the function f(lambda) = alpha * lambda^2 + beta * lambda
def objective(params, lambda_values):
    alpha, beta, gamma = params
    f_lambda = alpha * lambda_values**2 + beta * lambda_values
    max_lambda = lambda_values * (lambda_values > 0)
    return ((f_lambda - max_lambda)**2).sum()

# Initial guess for the parameters
initial_guess = [0, 0, 0]

# Range of lambda values
lambda_values = np.linspace(-1, 1, 100)

# Perform optimization
result = minimize(objective, initial_guess, args=(lambda_values,))
best_params = result.x

print("Best parameters found:", best_params)

# %%

import numpy as np
from scipy.optimize import minimize

# Define the function to minimize
def objective(a, x, y):
    return np.sum((np.maximum(x, 0) + np.maximum(y, 0)) - (a[0] * x**2 + a[1] * y**2 + a[2] * x * y + a[3] * x + a[4] * y + a[5]))**2

# Initial guess for the coefficients (a, b, c, d, e, f)
initial_guess = np.array([0.62, 0.62, 0, 0.5, 0.5, 0])

# Bounds for the coefficients (optional)
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

# Perform optimization
result = minimize(objective, initial_guess, args=(x, y))

# Optimal values for the coefficients
a = result.x

print("Optimal values for 'a':", a)

# %%

x = 1
y = 1

a[0]*x*x + a[1]*y*y + a[2]*x*y + a[3]*x + a[4]*y + a[5]

# %%

import numpy as np

# Define the binary matrix
and_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

or_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Diagonalize the matrix
diagonalized_matrix = np.diag(or_matrix)

print("Diagonalized Matrix:")
print(diagonalized_matrix)

# %%

x = 1
y = 0
z = 1

X = 0
Y = 2
Z = 3

data = [
    [-0.47178399562835693, 0.0011151726357638836, -0.4746299684047699, -0.4594239592552185, 0.7340422868728638],
    [0.0011151726357638836, -5.052223173152015e-07, 0.001119681866839528, 0.0010934143792837858, -0.00017485287389717996], 
    [-0.4746299684047699, 0.001119681866839528, -0.4774908125400543, -0.46220314502716064, 0.7368462085723877], 
    [-0.4594239592552185, 0.0010934143792837858, -0.46220314502716064, -0.44736164808273315, 0.7202721834182739], 
    [0.7340422868728638, -0.00017485287389717996, 0.7368462085723877, 0.7202721834182739, 0.00038151280023157597]
]

data[X][X]*x*x + 2*data[X][Y]*x*y + 2*data[X][Z]*x*z + 2*data[X][-1]*x + data[Y][Y]*y*y + 2*data[Y][Z]*y*z + 2*data[Y][-1]*y + data[Z][Z]*z*z + 2*data[Z][-1]*z + data[-1][-1]

# 0.09504442662000656 * x*x + 2*0.0940847247838974*x*y + 2*0.09508563578128815*x*z - 2*0.06488920748233795 * x + \
#      -0.05649254843592644 * y * y + 0.09420840442180634 * y * z + 2*0.010896424762904644 * y + \
#          0.09512683004140854 * z * z - 2*0.06495896726846695 * z
