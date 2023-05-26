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

        #this is to prevent the case where max_row_index is 0 (when there's no max, np.argmax returns 0)
        if max_row_index < i:
            max_row_index = i

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
    x_d = np.zeros(n, dtype=float)
    L, R =  lr_ohne(A)

    #forward substitution
    for i in range (n):
        for j in range (i):
            b[i] -= L[i][j] * x_d[j]
        x_d[i] = b[i]

    x = np.zeros(n, dtype=float)
    #backward substitution
    for i in reversed(range(n)):
        for j in range (n-i-1):
            #print(f"i: {i}, j: {j} ", end='') -> uncomment this line to understand how things workss
            x_d[i] -= R[i][n-j-1] * x[n-j-1]
        
        if(R[i][i] == 0):
            print ("Cannot solve this LGS")
            return
        
        x[i] = x_d[i] / R[i][i]
    return x

def linsolve_mit(A, b):
    n = len(A)
    x_d = np.zeros(n, dtype=float)
    L, R, P =  lr_mit(A)
    b = np.dot(b, P)
    #forward substitution
    for i in range (n):
        for j in range (i):
            b[i] -= L[i][j] * x_d[j]
        x_d[i] = b[i]

    x = np.zeros(n, dtype=float)
    #backward substitution
    for i in reversed(range(n)):
        for j in range (n-i-1):
            #print(f"i: {i}, j: {j} ", end='') -> uncomment this line to understand how things workss
            x_d[i] -= R[i][n-j-1] * x[n-j-1]
        
        if(R[i][i] == 0):
            print ("Cannot solve this LGS")
            return
        
        x[i] = x_d[i] / R[i][i]
    return x

def main():
    A = np.array([[1, 1, 3], [1, 2, 2], [2, 1, 5]])
    b = [2, 1, 1]

    x = linsolve_mit(A, b)

    print (x)


main()