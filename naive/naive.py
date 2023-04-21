import numpy as np
from numpy.linalg import inv
import test1
import test2
import sine

round_limit = 6
sortArray = True #not implemented yet




def naiv_inter(x, y, xi):
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
        return 2
    

    tempx = np.array(x)
    tempy = np.array(y)
    xy = np.ndarray(shape=(2,n), dtype=float)
    xy[0] = tempx
    xy[1] = tempy
    xpoints = np.array(xi)
    xmat = np.zeros((n,n))
    xmat.shape = (n,n)
    for i in range(n):
        for j in range(n):
            xmat[i,n-j-1] = xy[0][i] ** (j)
            #implement exception handling for identical tupels with different values
    #print(xmat)
    # invert xmat
    xmat = inv(xmat)
    #print(xmat)
    polynom = np.zeros(n)
    for i in range(n):
        temp_value = 0
        for j in range(n):
            temp_value += xy[1][j]*xmat[i][j]
        polynom[i] = np.round(temp_value, round_limit*2)
    
    #print this
    output_string = "Values of interpolation polynome I(x) = "
    for i in range(n):
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
    polynome(polynom, xi)


naiv_inter(test1.x, test1.y, test1.xi)
naiv_inter(test2.x, test2.y, test2.xi)
naiv_inter(sine.x,sine.y,sine.x)

