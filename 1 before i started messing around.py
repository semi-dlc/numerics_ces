import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time
import a1
import a2
#import sine



def naiv_inter_old(x, y, xi):
    size_M = len (x)
    Matrix = np.zeros ((size_M,size_M))
    for i in range (size_M):
        for j in range(size_M):
            Matrix[i][j] = pow (x[i], size_M-j-1)
    coeff = np.linalg.solve(Matrix, y)
    print(coeff)
    result = 0
    for i in range (len (coeff)):
        result += coeff[i] * pow(xi, len(coeff) - i - 1)
    return result

def naiv_inter(x, y, xi):
    round_limit = 6
    def polynome(p,x): #calculate polynom at given x points
            result = np.zeros(len(x))
            for i in range(len(x)):
                temp_val = 0
                for j in range(len(p)):
                    temp_val += p[len(p)-j-1] * x[i] ** j
                result[i] = np.round(temp_val, round_limit)
                print("Interpolation function at {point}: ".format(point=str(x[i])) + str(result[i]))
            return result
        
    n = len(x)
    if len(x) != len(y):
        print("x array (now of size {sizex}) and y array (now of size {sizey}) must be of the same size".format(sizex = len(x), sizey = len(y)))
        return -2
    

    xy = np.ndarray(shape=(2,n), dtype=float)
    xy[0] = np.array(x)
    xy[1] = np.array(y)
    xmat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            xmat[i,n-j-1] = xy[0][i] ** (j)
            #implement exception handling for identical tupels with different values
    # invert xmat
    xmat = inv(xmat)
    
    
    #create interpolation polynome
    polynom = np.zeros(n)
    for i in range(n):
        temp_value = 0
        for j in range(n):
            temp_value += xy[1][j]*xmat[i][j]
        polynom[i] = np.round(temp_value, round_limit*2)
    
    #print this
    def printpol(polynom):
        output_string = "Values of interpolation polynome I(x) = "
        for i in range(len(polynom)):
            output_string += str(polynom[i])
            output_string += "* x"
            if (n - i != 1 and n-i != 0):
                output_string += "^"
                output_string += str(n-i)
            output_string += " + "
        output_string += str(polynom[n-1])
        print(output_string)
        print("calculated at points:")
        print(xi)
   
    return polynome(polynom, xi)

def lagrange_inter (x, y, xi):
    result = 0
    for k in range (len (x)):
        l = 1
        for j in range (len (y)):
            if ( k != j):
                l = l * (xi - x[j]) / (x[k] - x [j])
        result += y[k] * l
    print("Results of lagrange:")
    print(result)
    while True:
        foo = input("Enter any key to exit")
        break
    return result

def divided_diff(x, y):
    n = len (y)
    coef = np.zeros([n, n])
    # the first column is y bc f[xn] = yn
    coef[:,0] = y
    # j represents the order of divided diff 
    # i represents the i in divided diff formula f[x_i ,..., x_j]
    # coef [i] [j] always comes from coef [i+1][j-1] - coef [i][j-1] / x[i+j] - x[i] (j-1 represents previous order of divided diff)
    for j in range(1,n):
        for i in range(n-j): #divided diff is represented thru diagonal matrix 
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i]) 
    print(coef)
    return coef

def newton_inter (x, y, xi):
    result = 0
    n = len (x)
    coef = divided_diff (x,y)
    for i in range (n):
        w = 1
        for j in range (i):
            w = w * (xi - x[j])
        result += coef[0, i] * w
    return result

def Fehler(f, fi):
    return np.abs(f - fi)

def main():
    ## Aufgabe 1 ##
    #a.
    #x = np.array([0,1,2,3])
    #y = np.array([-5,-6,-1,16])
    xi = np.array([0.5, 1.5, 2.5])
    
    print (naiv_inter(a1.x, a1.y , xi))
    print ("\n")
    print (naiv_inter_old(a1.x, a1.y , xi))
    print ("\n")
    print (lagrange_inter(a1.x, a1.y, xi))
    print ("\n")
    print (newton_inter(a1.x, a1.y, xi))

    #b
    #x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    #y = np.array([-5, -6, -1, 16, 10, 20, 9])
    xi = np.array([3.5,5])
    

    ## Aufgabe 2 ##
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    n = [5, 15 , 25]
    naiv_time = 0
    old_naiv_time = 0
    lagrange_time = 0
    newton_time  = 0
    for i, num_points in enumerate(n):
        xi = np.linspace(0, 2, num_points)
        yi = np.sin(xi)
        xi_new = np.linspace(0, 2, 200)

        # time naiv_inter function
        start_time = time.time()
        yi_naiv = naiv_inter(xi, yi, xi_new)
        naiv_time_n = time.time() - start_time
        print("Naive interpolation with n={} took {:.6f} seconds".format(num_points, naiv_time_n))
        naiv_time += naiv_time_n
        # time old naiv_inter function
        start_time = time.time()
        yi_old_naiv = naiv_inter_old(xi, yi, xi_new)
        old_naiv_time_n = time.time() - start_time
        print("Naive interpolation with n={} took {:.6f} seconds".format(num_points, naiv_time_n))
        old_naiv_time += old_naiv_time_n

        # time lagrange_inter function
        start_time = time.time()
        yi_lagrange = lagrange_inter(xi, yi, xi_new)
        lagrange_time_n = time.time() - start_time
        print("Lagrange interpolation with n={} took {:.6f} seconds".format(num_points, lagrange_time_n))
        lagrange_time += lagrange_time_n

        # time newton_inter function
        start_time = time.time()
        yi_newton = newton_inter(xi, yi, xi_new)
        newton_time_n = time.time() - start_time
        print("Newton interpolation with n={} took {:.6f} seconds".format(num_points, newton_time_n))
        newton_time += newton_time_n

        

        axs[0, i].plot(xi_new, np.sin(xi_new), color='g', label='sin(x)')
        axs[0, i].plot(xi_new, yi_old_naiv, color='r', label='Naive Interpolation')
        axs[0, i].plot(xi_new, yi_lagrange, color='y', label='Lagrange Interpolation')
        #axs[0, i].plot(xi_new, yi_newton, color='b', label='Newton Interpolation')
   
        axs[0, i].set_title(f'n={num_points}')
        axs[0, i].legend()

        # plot Fehler for each method
        Fehler_naiv = Fehler(np.sin(xi_new), yi_naiv)
        Fehler_lagrange = Fehler(np.sin(xi_new), yi_lagrange)
        Fehler_newton = Fehler(np.sin(xi_new), yi_newton)

       # axs[1, i].plot(xi_new, Fehler_naiv, label='Naive Interpolation Fehler')
        axs[1, i].plot(xi_new, Fehler_lagrange, label='Lagrange Interpolation Fehler')
        axs[1, i].plot(xi_new, Fehler_newton, label='Newton Interpolation Fehler')
        axs[1, i].set_title(f'n={num_points}')
        axs[1, i].legend()
    
    all_time  = [naiv_time/3, lagrange_time/3, newton_time/3]
    if min (all_time) == naiv_time/3:
        print ("Naiv inter used the least amount of time: {:.6f}" .format(min(all_time)))
    elif min (all_time) == lagrange_time/3:
        print ("Lagrange inter used the least amount of time: {:.6f}" .format(min(all_time)))
    else:
        print ("Newton method used the least amount of time: {:.6f}" .format(min(all_time)))
    plt.show()
    while True:
        foo = input("Enter any key to exit")
        return

main()