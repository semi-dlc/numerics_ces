import numpy as np
import matplotlib.pyplot as plt
import time

def naiv_inter(x, y, xi):
    size_M = len (x)
    Matrix = np.zeros ((size_M,size_M))
    for i in range (size_M):
        for j in range(size_M):
            Matrix[i][j] = pow (x[i], size_M-j-1)
    coeff = np.linalg.solve(Matrix, y)
    result = 0
    for i in range (len (coeff)):
        result += coeff[i] * pow(xi, len(coeff) - i - 1)
    return result

def lagrange_inter (x, y, xi):
    result = 0
    for k in range (len (x)):
        l = 1
        for j in range (len (y)):
            if ( k != j):
                l = l * (xi - x[j]) / (x[k] - x [j])
        result += y[k] * l
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

def Error(f, fi):
    return np.abs(f - fi)

def main():
    # Task 1 
    #a.
    x = np.array([0,1,2,3])
    y = np.array([-5,-6,-1,16])
    xi = np.array([0.5, 1.5, 2.5])
    
    print (naiv_inter(x, y , xi))
    print ("\n")
    print (lagrange_inter(x, y, xi))
    print ("\n")
    print (newton_inter(x, y, xi))

    #b
    x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    y = np.array([-5, -6, -1, 16, 10, 20, 9])
    xi = np.array([3.5,5])
    
    print (naiv_inter(x, y , xi))
    print ("\n")
    print (lagrange_inter(x, y, xi))
    print ("\n")
    print (newton_inter(x, y, xi))

    # Task 2
    fig, axs = plt.subplots(2, 3, figsize=(15,10))
    n = [5, 15 , 25]
    naiv_time = 0
    lagrange_time = 0
    newton_time  = 0
    for i, num_points in enumerate(n):
        xi = np.linspace(0, 2, num_points)
        yi = np.sin(xi)
        xi_new = np.linspace(0, 2, 1000)

        # time naiv_inter function
        start_time = time.time()
        yi_naiv = naiv_inter(xi, yi, xi_new)
        naiv_time_n = time.time() - start_time
        print("Naive interpolation with n={} took {:.6f} seconds".format(num_points, naiv_time_n))
        naiv_time += naiv_time_n
        

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

     

        axs[0, i].plot(xi_new, yi_naiv, label='Naive Interpolation')
        axs[0, i].plot(xi_new, yi_lagrange, label='Lagrange Interpolation')
        axs[0, i].plot(xi_new, yi_newton, label='Newton Interpolation')
        axs[0, i].plot(xi_new, np.sin(xi_new), label='sin(x), Original function')
        axs[0, i].set_title(f'n={num_points}')
        axs[0, i].legend()

        # plot Error for each method
        Error_naiv = Error(np.sin(xi_new), yi_naiv)
        Error_lagrange = Error(np.sin(xi_new), yi_lagrange)
        Error_newton = Error(np.sin(xi_new), yi_newton)

        axs[1, i].plot(xi_new, Error_naiv, label='Naive Interpolation Error')
        axs[1, i].plot(xi_new, Error_lagrange, label='Lagrange Interpolation Error')
        axs[1, i].plot(xi_new, Error_newton, label='Newton Interpolation Error')
        #note that error for n= 25 is higher than n = 15; inaccuracy of float-arithmetic?
        axs[1, i].set_title(f'n={num_points}')
        axs[1, i].legend()
    
        all_time  = [naiv_time, lagrange_time, newton_time]
        if min (all_time) == naiv_time:
            print ("Naiv inter used the least amount of time: {:.6f} seconds" .format(min(all_time)))
        elif min (all_time) == lagrange_time:
            print ("Lagrange inter used the least amount of time: {:.6f} seconds" .format(min(all_time)))
        else:
            print ("Newton method used the least amount of time: {:.6f} seconds" .format(min(all_time)))
        print ("\n")
    plt.show()
    while(True):
        foo = input("Enter any character to exit")
        if (foo != ""):
            return
    
main()
