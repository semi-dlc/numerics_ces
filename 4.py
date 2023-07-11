import numpy as np
import matplotlib.pyplot as plt
import random 
import math

def f1(x):
    #fx1 = x[0]**2 + x[1]**2 - 1
    #fx2 = x[0]*x[1] - 1/4
    return np.array([x[0]**2 + x[1]**2 - 1, x[0]*x[1] - 1/4])
#roots are sign(sqrt(0.5+sqrt(3)/4))



def Df1 (x):
#    Df1x = 2*x[0]
 #   Df1y = 2*x[1]
  #  Df2x = x[1]
   # Df2y = x[0]
    return np.array([[2*x[0], 2*x[1]],[x[1], x[0]] ])

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

def f3(x):
#   xx = x[0] / 2 * np.sin(np.pi * x[0]) - x[1]
#   yy = x[1] ** 2 - x[0] + 1
    return np.array([x[0] / 2 * np.sin(np.pi * x[0]) - x[1], x[1] ** 2 - x[0] + 1])
def Df3(x):
 #   Dx0 = 
  #  Dx1 = -1
#    Dy0 = -1
#    Dy1 = 2*x[1]
    return np.array([[0.5 * np.sin(np.pi * x[0] / 2) + 0.25 * x[0] * np.pi * np.cos(np.pi * x[0] / 2), -1],[-1, 2*x[1]]])


def f4 (x):
    return (x**2 - 4) * (x**2 - 1)

#minima of f4 are sqrt(2.5),so m1,m2 = -1.58, 1.58
#maximum is 0
#roots are -2 -1 1 2

def Df4(x):
    return 4*x**3-10*x

def f5(x, j_d):
    z = x[0] + x[1] * 1j 
    f = z**j_d - 1
    return np.array([f.real, f.imag])

def zj(j_d): ##root of f5j
    z = np.zeros((j_d, 2))
    for k in range (1, j_d+1):
        theta = 2 * np.pi * k / j_d
        zk = np.array([np.cos(theta), np.sin(theta)]) #Careful! Because np.pi is not exactly pi, there are some numerical errors that might become relevant
        z[k-1] = zk
    return z

def Df5(x, j_d):
    #a = x[0]
    #b = x[1]
    z = x[0] + x[1] * 1j
    df_da = j_d * z**(j_d-1)
    df_db = j_d * z**(j_d-1) * 1j
    
    return np.array([[df_da.real, df_db.real], [df_da.imag, df_db.imag]])

## TBD: partial derivative of gj
    #We need to first analyze how this works.
    #Ok now I know. Thx...
## TBD: Jacobian matrix of f5,j
    

def error (x1, x0):
    return np.linalg.norm(x1-x0, ord=2)

def error_1D (x1,x0):
    return np.abs(x1 - x0)

def error_C(x, zj):
    error_list = []
    for zk in zj:
        error = np.linalg.norm(x-zk, ord=2)
        error_list.append(error)
    error_min = min (error_list)
    return error_min

## to do for fun.. (?) do a function that is a map -> so Df can be a function applicable to all functions from f1-f3
## right now newton works fine
## in place how? One possiblity is to save the old value in temp but u still need the origianl x to input it in Df tho


def newton_old_insecure (x0, f, Df, tol, itmax, x = []):
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

def newton (x0, f, Df, tol, itmax, x = []): #this one is used
    Singular_Count = 0 #if too many singular matrices, return error
    x = []
    x.append(x0)
    print(f(x0))
    print(Df(x0))
    try:
        x_new  = x[0] - np.linalg.inv(Df(x[0])).dot(f(x[0]))
    except np.linalg.LinAlgError: # be x a vector of real numbers
        epsilonJacobian = 0.01

        print("Jacobian is singular. x gets a slight offset")
        for i in range(len(x[0])):
            x[0][i] = x[0][i] + i * 0.01
            print(x[0][i])
        print(x[0])
        x_new  = x[0] - np.linalg.inv(Df(x[0])).dot(f(x[0]))
    x.append(x_new)
    itmax = 0
    while error(x[itmax+1], x[itmax]) > tol * np.linalg.norm(x[itmax], ord=2):
        try:
            x_new  = x[itmax+1] - np.linalg.inv(Df(x[itmax+1])).dot(f(x[itmax+1]))
        except np.linalg.LinAlgError: # be x a vector of real numbers
            Singular_Count += 1
            epsilonJacobian = 0.01
            print("Jacobian is singular. x gets an offset")
            if Singular_Count > 7:
                print("Newton iteration might be stuck in a loop. The Jacobian is regularly singular")
                return x, itmax
            for i in range(len(x[0])):
                x[itmax + 1][i] = x[itmax + 1][i] + i
                print(x[itmax + 1][i])
            print(x[itmax + 1])
            x_new  = x[itmax + 1] - np.linalg.inv(Df(x[itmax + 1])).dot(f(x[itmax + 1]))
        x.append(x_new)
        itmax += 1
    return x, itmax


def newton_C (x0, j, f, Df, tol, itmax, x = []):
    x = []
    x.append(x0)
    try:
        x_new  = x[0] - np.linalg.inv(Df(x[0], j)).dot(f(x[0], j))
    except :
        x_new = [math.nan, math.nan]
        x.append(x_new)
        return x, itmax
    x.append(x_new)
    itmax = 0

    ## idea ## find out each time which element of zj is closest to the current x[itmax] using min {}
    while error_C(x[itmax], zj(j)) >= tol: ## to be worked on. The condition here is different and has sth to do with zj
        try:
            x_new  = x[itmax+1] - np.linalg.inv(Df(x[itmax+1], j)).dot(f(x[itmax+1],j))
        except :
            x_new = [math.nan, math.nan]
            x.append(x_new)
            itmax += 1
            return x, itmax
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

## to do for fun.. (?) do a function that is a map -> so Df can be a function applicable to all functions from f1-f3
## right now newton works fine

##############fun by dino

def log_transform(l):
    return (2**(l+1)) / (2**l + 1) - 1
def simple_transform(l):  #faster to compute
    return l / (abs(l) + 1)
def plot_R2(x_it, fx): # be x_it domain, and fx image
    stepbystep = False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    for i in range(len(x_it)):
        #coord = [x_it[i][0], x_it[i][1], fx[i][0]]
    #print(coord)
     #   print(fx[i][1])
     #   print("Color (normalized)")
        color__ = simple_transform(fx[i][1]) 
      #  print(color__)
        if stepbystep:
            while (input() == None): #step by step iteration
                pass
        if color__ >= 0:
            ax.scatter(x_it[i][0], x_it[i][1], fx[i][0], c = [(1-color__, 1, 1)], edgecolors = (0,0,0))
        else: 
            ax.scatter(x_it[i][0], x_it[i][1], fx[i][0], c = [(1, 1-abs(color__), 1)], edgecolors= (0, 0, 0))
        fig.show() 
    fig.show()
    s = None
    while (input("Press to continue") == None):
        pass


def wrapper(fn, Dfn, starting_point = np.array([1,1])):
    itmax = 2
    x1, itmax = newton(starting_point, fn, Dfn, 0.0000000000001, itmax)
    print (x1)
    print (itmax)
    fx1 = []
    for i in x1:
        fx1.append(f1(i))
    plot_R2(x1, fx1)    	    




def main():
    ## partially a.) ##
    ## TBD !! ##
    '''
    x1_0 = np.array ([1,2])
    wrapper(f1, Df1, x1_0)

    x2_0 = np.array([1, -3])
    wrapper(f2, Df2, x2_0)
    
    x3_0 = np.array([1, -2])
    wrapper(f3, Df3, x3_0)


    ## b.) ##
    ## generate 1000 in [-2.5, 2.5]
    n4 = 10000
    x4_0 = np.linspace(-2.5, 2.5, n4)
    x4 = np.zeros_like(x4_0)
    it4 = np.zeros_like(x4_0)
    for i in range (0,n4):
        x4[i], it4[i] = newton_1D(x4_0[i], f4, Df4, 0.00000001, it4[i])
    
    print(x4)
    print (it4)

    plt.plot(x4_0, it4, 'b.', label='x0 from [-2.5,2.5]', markersize=2)
    #[-2.5,-2] converges to -2: black
    plt.plot(x4_0[(x4_0 >= -2.5) & (x4_0 <= -2)], it4[(x4_0 >= -2.5) & (x4_0 <= -2)], 'k.', label='x0 from [-2.5,-2]', markersize=2)
    #[-1.1, -0.9] converges to -1: red
    plt.plot(x4_0[(x4_0 >= -1.1) & (x4_0 <= -0.9)], it4[(x4_0 >= -1.1) & (x4_0 <= -0.9)], 'r.', label='x0 from [-1.1, -0.9]', markersize=2)
    #[0.9, 1.1] converges to -1: green
    plt.plot(x4_0[(x4_0 >= 0.9) & (x4_0 <= 1.1)], it4[(x4_0 >= 0.9) & (x4_0 <= 1.1)], 'g.', label='x0 from [0.9, 1.1]', markersize=2)
    #[2,2.5] converges to 2: yellow
    plt.plot(x4_0[(x4_0 >= 2) & (x4_0 <= 2.5)], it4[(x4_0 >= 2) & (x4_0 <= 2.5)], 'y.', label='x0 from [2,2.5]', markersize=2)
    plt.xlabel('x0')
    plt.ylabel('iterations')
    plt.title('Newton Iterations')
    plt.legend()
    plt.show()
    
    '''
    j = 5

    min_re = -1
    max_re = 1
    min_im = -1
    max_im = 1

    num_points = 100


    # Create the grid of points
    x = np.linspace(min_re, max_re, num_points)
    y = np.linspace(min_im, max_im, num_points)
    X, Y = np.meshgrid(x, y)

    x5_0 = np.vstack([X.flatten(), Y.flatten()]).T
    x5 = np.zeros_like(x5_0)
    it5 = np.zeros_like(x5_0)
    fractal_map = np.zeros((num_points, num_points))
    roots = zj(j)
    for i in range (0, num_points):
        for k in range (0, num_points):
            comp = np.array([0,0])
            comp, it5[i] = newton_C(x5_0[i], j, f5, Df5, 0.000000001, it5[i])
  #          print (comp)         
            for l in range(0, len(roots), 1):
                if error(roots[l], comp) / np.linalg.norm(roots[l], ord = 2) < 0.01:
                    fractal_map[i][k] = l + 1
    print(fractal_map)
    plt.imshow(fractal_map)
    plt.show()
    print (zj(j))

    
main()

