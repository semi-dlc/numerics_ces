import numpy as np
import matplotlib.pyplot as plt

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

def f4 (x):
    return (x**2 - 4) * (x**2 - 1)

def Df4(x):
    return 4*x**3-10*x

def gj(z, j):
    return z**j - 1

def zkj(j, k):
    theta = 2 * np.pi * k / j
    z = complex(np.cos(theta), np.sin(theta))
    result = gj(z, j)
    return np.real(result), np.imag(result)

## TBD: partial derivative of gj
    #We need to first analyze how this works.
    #I don't really understand much. But we can use binomial coeff and try to write this down on paper first.
## TBD: Jacobian matrix of f5,j
    

def error (x1, x0):
    return np.linalg.norm(x1-x0, ord=2)

def error_1D (x1,x0):
    return np.abs(x1 - x0)

## to do for fun.. (?) do a function that is a map -> so Df can be a function applicable to all functions from f1-f3
## right now newton works fine
## in place how? One possiblity is to save the old value in temp but u still need the origianl x to input it in Df tho


def newton (x0, f, Df, tol, itmax, x = []):
    x = []
    x.append(x0)
    x_new  = x[0] - np.linalg.inv(Df(x[0])).dot(f(x[0]))
    x.append(x_new)
    itmax = 0
    while error(x[itmax+1], x[itmax]) > tol * np.linalg.norm(x[itmax], ord=2):
        x_new  = x[itmax+1] - np.linalg.inv(Df(x[itmax+1])).dot(f(x[itmax+1]))
        x.append(x_new)
        itmax += 1
    return x[itmax], itmax

## to be implemented to just be in newton()
def newton_1D(x0, f, Df, tol, it_max, x=[]):
    x = []
    x.append(x0)
    x_new = x[0] - f(x[0]) / Df(x[0])
    x.append(x_new)
    itmax = 0
    while error_1D(x[itmax+1], x[itmax]) > tol * abs(x[itmax]):
        x_new = x[itmax+1] - f(x[itmax+1]) / Df(x[itmax+1])
        x.append(x_new)
        itmax += 1
    return x[itmax], itmax


def main():
    ## partially a.) ##
    ## TBD !! ##
    x1_0 = np.array ([1,2])
    itmax = 2
    x1, itmax = newton(x1_0, f1, Df1, 0.0000000000001, itmax)
    print (x1)
    print (itmax)


    ## b.) ##
    ## generate 1000 in [-2.5, 2.5]
    n = 1000
    x4_0 = np.linspace(-2.5, 2.5, n)
    x4 = np.zeros_like(x4_0)
    it4 = np.zeros_like(x4_0)
    for i in range (0,n):
        x4[i], it4[i] = newton_1D(x4_0[i], f4, Df4, 0.00000001, it4[i])
    
    print(x4)
    print (it4)

    plt.plot(x4, it4, color='blue', label='x0 from [-2.5,2.5]')
    #[-2.5,-2] converges to -2: black
    plt.plot(x4[(x4 >= -2.5) & (x4 <= -2)], it4[(x4 >= -2.5) & (x4 <= -2)], color='black', label='x0 from [-2.5,-2]')
    #[-1.1, -0.9] converges to -1: red
    plt.plot(x4[(x4 >= -1.1) & (x4 <= -0.9)], it4[(x4 >= -1.1) & (x4 <= -0.9)], color='red', label='x0 from [-1.1, -0.9]')
    #[0.9, 1.1] converges to -1: green
    plt.plot(x4[(x4 >= 0.9) & (x4 <= 1.1)], it4[(x4 >= 0.9) & (x4 <= 1.1)], color='green', label='x0 from [0.9, 1.1]')
    #[2,2.5] converges to 2: orange
    plt.plot(x4[(x4 >= 2) & (x4 <= 2.5)], it4[(x4 >= 2) & (x4 <= 2.5)], color='orange', label='x0 from [2,2.5]')
    plt.xlabel('x0')
    plt.ylabel('iterations')
    plt.title('Newton Iterations')
    plt.legend()
    plt.show()

    #Yes it's beautiful
    # The problem with this is that, plot tries to connect the dot and the blue line doesn't make much sense.

main()

    # This is so beautiful guys ToT. Lmao, FOR REAL WITH ALL THESE COMMENTS AND PPL STILL SAY I USE CHATGPT.

main()
