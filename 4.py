import numpy as np

def f1(x):
    fx1 = x[0]**2 + x[1]**2 - 1
    fx2 = x[0]*x[1] - 1/4
    return np.array([fx1, fx2])

def Df1 (x):
    Df1x = 2*x[0]
    Df1y = 2*x[1]
    Df2x = x[1]
    Df2y = x[0]
    return np.array([[Df1x, Df1y], [Df2x, Df2y]])

def error (x1, x0):
    return np.linalg.norm(x1-x0, ord=2)

## to do for fun.. (?) do a function that is a map -> so Df can be a function applicable to all functions from f1-f3
## right now newton works fine
## in place how? One possiblity is to save the old value in temp but u still need the origianl x to input it in Df tho


def newton (x0, f, Df, tol, itmax, x = []):
    x.append(x0)
    x_new  = x[0] - np.linalg.inv(Df(x[0])).dot(f(x[0]))
    x.append(x_new)
    itmax = 0
    while error(x[itmax+1], x[itmax]) > tol * np.linalg.norm(x[itmax], ord=2):
        x_new  = x[itmax+1] - np.linalg.inv(Df(x[itmax+1])).dot(f(x[itmax+1]))
        x.append(x_new)
        itmax += 1

    return x[itmax], itmax

def main():
    x0 = np.array ([1,2])
    itmax = 0
    x, itmax = newton(x0, f1, Df1, 0.0000000000001, itmax)
    print (x)
    print (itmax)

main()