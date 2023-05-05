import numpy as np
import math
import matplotlib.pyplot as plt
import time

def  gaussq_n(f,a,b,n):
    A = np.zeros ((n,n))
    for i in range (1, n):
        A[i-1][i] = A[i][i-1] = i/ np.sqrt(4 * pow(i,2) - 1)
    eig_vals, eig_vecs = np.linalg.eigh(A)
    x = eig_vals
    w = np.array(n)
    w = 2 * pow (eig_vecs[0],2)
    #u-Transformation
    if a != -1 or b!= 1:
        w = w * (b-a)/2 
        x = x* (b-a)/2 + (b+a)/2 
    result = 0
    for i in range (n):
        result += w[i] * f(x[i])

    return result
def midpoint_rule(f, a, b):
    return f((a+b)/2)*(b-a)

def open_trapez_rule (f,a,b):
    return (f((2*a+b)/3) + f((a+2*b)/3))* (b-a)/2

def gaussq_tol(f,a,b,tol):
    Q = [gaussq_n(f,a,b,2), gaussq_n(f,a,b,3)] #bc gaussq_n(f,a,b,0) and gaussq_n(f,a,b,0) return 0
    while np.abs (Q[len(Q)-1] - Q[len(Q)-2]) > tol:
        Q.append(gaussq_n(f,a,b,len(Q)))
    return len(Q)




def f_a(x):
    return pow(x,10)

def f_c(x):
    return 1/ (0.01 + pow(x,2))

def main():
    print (gaussq_tol(f_a, -1,1, 0.0001))
    print (gaussq_n(f_a,-1,1,gaussq_tol(f_a, -1,1, 0.0001)))

    print (gaussq_tol(math.sin, 0, math.pi , 0.0001))
    print (gaussq_n(math.sin, 0, math.pi , gaussq_tol(math.sin, 0, math.pi , 0.0001)))

    print (gaussq_tol(f_c, -2, 3, 0.0001))
    print (gaussq_n(f_c, -2, 3 , gaussq_tol(f_c, -2, 3, 0.0001)))
main()
