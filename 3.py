import numpy as np

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
    epsilon = np.random.normal(0, 1)

    array_of_matrices = []
    for n in range(2, 7):
        matrix = np.zeros((n, n))
        for i in range (1, n+1):
            for j in range (1, n+1):
                matrix [i-1][n-j] = pow (3, -abs(i-j)) + pow (2, -j-i) + pow (10,-10) * epsilon
        array_of_matrices.append(matrix)
    

    ## d. ##
    n =3  #I think we have to change n and plot here UwU -> like if n increase what happens????
    H = np.zeros((n,n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            H[i-1][j-1] = 1 / (i + j - 1)

    x_d = np.ones(n)
    b = np.dot(H, x_d)

    #x_h2 = linsolve_ohne(H, b) #If I uncomment this the result of linsolve_ohne below will be different like what???

    print (np.linalg.solve(H, b))
    print (linsolve_ohne (H, b))
    print (linsolve_mit (H, b))
    
    ##I tried to optimize the code by not having multiple set of x. However, I still copy A -> R because unless I do that, the value
    #of A change and I won't be able to produce an accurate result for the next linsolve function

main()
