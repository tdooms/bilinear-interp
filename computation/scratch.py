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
