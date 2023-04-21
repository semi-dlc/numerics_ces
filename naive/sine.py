import numpy as np


xlow = 0
xup = 2

nn = input("Amount of interpolation nodes:")
nn = int(nn)
#nn = 2/(nn-1) #not necessary as we use linspace
x = np.linspace(xlow, xup, num = nn)
y = np.sin(x)
