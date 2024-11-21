# %%
import numpy as np

a = 0.30109, 0.6627
b = 0.2652, 0.584
c = 0.2523, 0.5533

import matplotlib.pyplot as plt

# Points
points = np.array([a, b, c])

# Extract x and y coordinates
x = points[:, 0]
y = points[:, 1]

# Perform linear fit without intercept
coefficients = np.polyfit(x, y, 1, full=True)[0]
slope = coefficients[0]

# Print the slope
print("Slope:", slope)

# Plot the points and the linear fit
plt.scatter(x, y, color='red', label='Data points')
plt.plot(x, slope * x, label='Linear fit without intercept')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
