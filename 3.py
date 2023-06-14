import numpy as np
import math
import matplotlib.pyplot as plt

def lr_ohne(A):
    n = len(A)
    L = np.eye(n, dtype=float)
    R = np.copy(A).astype(float)
    P = np.eye(n)
    
    for i in range(0, n-1):
        for j in range(i+1, n):
            if R[i][i] == 0:
                l = 0
            else:
                l = R[j][i] / R[i][i]
            L[j][i] = l
            R[j] = R[j] - l*R[i]
    return L, R

def lr_mit (A):
    n = len(A)
    L = np.eye(n, dtype=float)
    R = np.copy(A).astype(float)
    P = np.eye(n)
    
    for i in range(0, n-1):
        column_values = R[i:, i]  # Get the values in the specified column
        max_row_index = np.argmax(column_values)  # Find the index of the maximum value
        if (abs(R[np.argmin(column_values)][i]) > R[np.argmax(column_values)][i]): #find index of absolute value
            max_row_index = np.argmin(column_values)

        #this is to prevent the case where max_row_index is 0 (when there's no max, np.argmax returns 0)
        if max_row_index < i:
            max_row_index = i

       ### print("Maximalspaltenindex")
       ### print(max_row_index)
        ###print("von spalte")
       ### print(column_values)
        #Row swap
        R[[i, max_row_index]] = R[[max_row_index, i]]
        L[[i, max_row_index], :i] = L[[max_row_index, i], :i]
        P[[i, max_row_index]] = P[[max_row_index, i]]
        for j in range(i+1, n):
            if R[i][i] == 0:
                l = 0
            else:
                l = R[j][i] / R[i][i]
            L[j][i] = l
            R[j] = R[j] - l*R[i]
    return L, R, P
    
def linsolve_ohne(A, b):
    n = len(A)
    x = np.zeros(n, dtype=float)
    L, R = lr_ohne(A)
    b_use = np.copy (b)
    # forward substitution
    for i in range(n):
        if R[i][i] == 0:
            print("Cannot solve this LGS")
            return
        for j in range(i):
            b_use[i] -= L[i][j] * x[j]
        x[i] = b_use[i]
    # backward substitution
    for i in reversed(range(n)):
        for j in range (n-i-1):
            #print(f"i: {i}, j: {j} ", end='') -> uncomment this line to understand how things works
            x[i] -= R[i][n-j-1] * x[n-j-1]
        
        
        x[i] /= R[i][i]
    return x




def linsolve_mit(A, b):
    n = len(A)
    x = np.zeros(n, dtype=float)
    L, R, P =  lr_mit(A)
    b_use = np.dot(P,b)
    #forward substitution
    for i in range (n):
        for j in range (i):
            b_use[i] -= L[i][j] * x[j]
        x[i] = b_use[i]
    

    #backward substitution
    for i in range(n-1, -1, -1):
        x[i] = x[i]
        for j in range(n-1, i, -1):
            #print(f"i: {i}, j: {j} ", end='') -> uncomment this line to understand how things works
            x[i] -= R[i][j] * x[j]
    
        if R[i][i] == 0:
            print("Cannot solve this LGS")
            return np.nan
        
        x[i] /= R[i][i]

    return x

def error_rel(v1, v2): #norm and pow may be subject to rounding errors
    if (len(v1) != len(v2)):
        print("Sizes don't match. Comparison is not possible")
        return math.inf
    ret = pow(np.linalg.norm(v1-v2),2)/pow(np.linalg.norm(v1), 2)
    return ret
    

def main():

    ## a. ##
    A = np.array([[1, 1, 3], [1, 2, 2], [2, 1, 5]])
    AA = np.array([[-11, 1, 3], [-2, -22, 2], [2, 1, -5]])
    b = [2, 1, 1]
    
    #print (linsolve_mit(AA,b))
    print (linsolve_ohne(A,b))
    print (linsolve_mit(A,b))
    
    
    ## b. ##

    A = np.array([[0, 1, 0, -1],
              [1, 2, 2, 1],
              [1, 1, 1, 1],
              [2, 1, -1, 2]])

    b = np.array([2, -1, -2, -11])

    print (linsolve_ohne (A,b)) #Cannot solve this LGS ->
    #For some strange reasons, after linsolve_ohne is called, the last member of b was incremented, making b = 2,-1,-2,-10
    #So I put it on comment, but u guys can uncomment it and see for urself what's up here
    print (linsolve_mit(A,b))

    ## c. ## -> Im not gonna do this lol. I don't really understand what to do.

    np.random.seed(42)
    

    array_of_matrices = []
    array_of_vectors = []
    array_of_errors_ohne = []
    array_of_errors_mit = []
    nstart = 2
    nend = 7
    epsilon_strength = -10
    for n in range(nstart, nend):
        matrix = np.zeros((n, n))
        for i in range (1, n+1):
            for j in range (1, n+1):
                epsilon = np.random.normal(0, 1)
                matrix [i-1][n-j] = pow (3, -abs(i-j)) + pow (2, -j-i) + pow (10,epsilon_strength) * epsilon 
        #print(matrix)
        x = np.ones(n)
        array_of_vectors.append(np.matmul(matrix, x))
        #print(np.matmul(matrix, x))
        array_of_matrices.append(matrix)

    print("Solving : ")
    fig1, axs1 = plt.subplots(3, 1)    
    
    for i in range(len(array_of_matrices)): # Solution should be very close to ones(i)
        sol = np.ones(i+2)
        print("Size " + str(i+2))
        print("Pivot")
        print(linsolve_mit(array_of_matrices[i], array_of_vectors[i]))
        print("non-Pivot")
        print(linsolve_ohne(array_of_matrices[i], array_of_vectors[i]))
        print("numpy")
        print(np.linalg.solve(array_of_matrices[i], array_of_vectors[i])) #could this be slightly inaccurate?

        #one more variable for better readablity and ensuring that one line of code is able to fit into one line of IDE

        error_temp = 0
        print("Error non-pivot")
        error_temp = error_rel(linsolve_ohne(array_of_matrices[i], array_of_vectors[i]), np.linalg.solve(array_of_matrices[i], array_of_vectors[i]))
        array_of_errors_ohne.append(error_temp)
        print(error_temp)

        print("Error pivot")
        error_temp = error_rel(linsolve_mit(array_of_matrices[i], array_of_vectors[i]), np.linalg.solve(array_of_matrices[i], array_of_vectors[i]))
        array_of_errors_mit.append(error_temp)
        print(error_temp)

    #plot part
    axs1[0].plot(np.arange(nstart, nend), array_of_errors_ohne, 'r-o', label = "Error without pivot")
    axs1[0].set_xlabel('Size of matrix')
    axs1[0].set_ylabel('Error')
    axs1[0].set_title("Error without pivot")

    axs1[1].plot(np.arange(nstart, nend ), array_of_errors_mit, 'g-o', label = "Error with pivot")
    axs1[1].set_xlabel('Size of matrix')
    axs1[1].set_ylabel('Error')
    axs1[1].set_title("Error with pivot")

    axs1[2].plot(np.arange(nstart, nend ), array_of_errors_ohne, 'r-o', label = "non-pivot")
    axs1[2].plot(np.arange(nstart, nend ), array_of_errors_mit, 'g-o', label = "pivot")
    axs1[2].set_ylabel('Error')
    axs1[2].set_xlabel('Size of matrix')
    axs1[2].legend()
    axs1[2].set_title("Errors compared to each other")
    fig1.set_size_inches(12, 18)
    plt.draw() 

    ## d. ##
    nstart = 3  
    nend = 10
    array_of_errors_ohne_h = []
    array_of_errors_mit_h = []
    for n in range(nstart, nend + 1) :   
        H = np.zeros((n,n))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                H[i-1][j-1] = 1 / (i + j - 1)

        x_d = np.ones(n)
        b = np.dot(H, x_d)

    #x_h2 = linsolve_ohne(H, b) #If I uncomment this the result of linsolve_ohne below will be different like what???

        #print (np.linalg.solve(H, b))
        #print (linsolve_ohne (H, b))
        #print (linsolve_mit (H, b))
        error_temp = 0

        error_temp = error_rel(linsolve_ohne(H, b), np.ones(n))
        array_of_errors_ohne_h.append(error_temp)
       

        error_temp = error_rel(linsolve_mit(H, b), np.ones(n))
        array_of_errors_mit_h.append(error_temp)

    fig2, axs2 = plt.subplots(3, 1)    
    #plot part
    axs2[0].plot(np.arange(nstart, nend+1), array_of_errors_ohne_h, 'r-o', label = "Error without pivoting")
    axs2[0].set_xlabel('Size of hilbert matrix')
    axs2[0].set_ylabel('Error')
    axs2[0].set_title("Error without pivoting")

    axs2[1].plot(np.arange(nstart, nend+1 ), array_of_errors_mit_h, 'g-o', label = "Error with pivoting")
    axs2[1].set_xlabel('Size of hilbert matrix')
    axs2[1].set_ylabel('Error')
    axs2[1].set_title("Error with pivot")

    axs2[2].plot(np.arange(nstart, nend+1 ), array_of_errors_ohne_h, 'r-o', label = "non-pivot")
    axs2[2].plot(np.arange(nstart, nend+1), array_of_errors_mit_h, 'g-o', label = "pivot")
    axs2[2].set_ylabel('Error')
    axs2[2].set_xlabel('Size of hilbert matrix')
    axs2[2].legend()

    axs2[2].set_title("Errors compared to each other")

    fig2.set_size_inches(12, 18)        

    #interestingly enough, pivotization and nonpivotization return same error margins.
    #this is because of the nature of the hilbert matrix, where the biggest value is a11.
    
    ##warisa tried to optimize the code by not having multiple set of x. However, warisa still copy A -> R because unless I do that, the value
    #of A change and warisa won't be able to produce an accurate result for the next linsolve function
    plt.show()

main()
