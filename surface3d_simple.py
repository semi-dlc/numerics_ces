import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f2(x):
  #  xx = x[0]**2 - x[1] - 2
  #  yy = x[0] * x[1] + 1
    return np.array([x[0]**2 - x[1] - 2, x[0] * x[1] + 1])
def Df2(x):
 #   Dx0 = 2*x[0]
 #   Dx1 = x[1]
 #   Dy0 = -1
 #   Dy1 = x[0]
    return np.array([[2*x[0], x[1]], [-1, x[0]]])

def log_transform(l):
    return (2**l) / (2**l + 1)
# Generate the input values for the x and y axes using a meshgrid

xmin = -10
xmax = 10
step = 0.1
x = np.arange(xmin, xmax, step)
y = np.arange(xmin, xmax, step)
X, Y = np.meshgrid(x, y)
xy = np.array([X,Y])
# Calculate the output values by calling your function with the input values
Z = f2(xy)
print(Z)
# Create an external array for coloring
print(Z[1])
color_array = log_transform(Z[1]) # example array, modify as needed #this needs a logistic transformation
print(color_array)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with colors based on the external array
surf = ax.plot_surface(X, Y, Z[0], facecolors=plt.cm.seismic(color_array)) #roots are white

# Add a colorbar for reference
#fig.colorbar(seismic)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Output')
ax.set_title('3D Function Plot')

plt.show()