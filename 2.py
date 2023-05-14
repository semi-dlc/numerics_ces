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

def sumtrap_closed(f,a,b,n): #closed trapez rule. n is amount of degrees (=quadrature points)
    ret = 0
    width = abs(b-a)/(n-1)
    print("width: ", width)
    for i in np.arange(0, b-a, width):
        add = f(a+i)*width + 0.5 * width * (f(a+i+width) - f(a+i))
       # print(add, a+i, f(a+i), f(a+i+width))
        ret += add
        
    return ret

def gaussq_tol(f,a,b,tol):
    Q = [gaussq_n(f,a,b,2), gaussq_n(f,a,b,3)] #bc gaussq_n(f,a,b,0) and gaussq_n(f,a,b,0) return 0
    while np.abs (Q[len(Q)-1] - Q[len(Q)-2]) > tol:
        Q.append(gaussq_n(f,a,b,len(Q)))
    return len(Q)

def test_int(f,a,b,n):
    print("Gauss {%f}", gaussq_n(f,a,b,gaussq_tol(f,a,b,n)))
    print("Degree was {%u}", gaussq_tol(f, a,b, n))
    #print("midpoint rule {%f}",  midpoint_rule(f, a, b))
    print("Trapez, {%f}", open_trapez_rule (f,a,b))
    print("Summed Trapez, closed {%f}", sumtrap_closed(f,a,b, gaussq_tol(f,a,b,n)))
    print("Success")
    return

def f_a(x):
    return pow(x,10)

def f_c(x):
    return 1/ (0.01 + pow(x,2))


def error(f, integral, a, b, nmax, method): #integral is value of integral from  a to b
    ni = np.arange(1,nmax)
    errors = np.zeros(nmax)
    for i in ni:
        errors[i-1] = abs(integral - method(f, a, b, i))
    print(errors)
    return errors
def plotting_init(f, integral, a, b, nmax): #this is an inefficient approach
    fig, axs = plt.subplots()
    axs.plot(np.arange(0,nmax,1), error(f, integral, a,b,nmax, gaussq_n), label = "Gauss error") 
    axs.plot(np.arange(0,nmax,1), error(f, integral, a, b, nmax, sumtrap_closed), label = "Trapez error") 
    plt.xlabel('Amount of quadrature points (Stützstellen)')
    plt.ylabel('Error')
    plt.title("Error plot")
    plt.legend()
    plt.show()
    return

def plotting_tol(f, a, b, tol_min_exponent): #min_exponent: Smallest expoent, for example 10^-5 -> -5
    tol_range = np.power(10, np.arange(tol_min_exponent))
    y_out = np.zeros(abs(tol_min_exponent))
    for iter in tol_range:
        y_out[iter] = gaussq_tol(f, a, b, iter)
    print(y_out)
    return
    

def main():
    #constants
    int_a = 2/11
    int_b = 2
    int_c = 30.58313262

    n_in = 20

    tol = 0.0001
    #print (gaussq_tol(f_a, -1,1, 0.0001))
    #print (gaussq_n(f_a,-1,1,gaussq_tol(f_a, -1,1, 0.0001)))
    test_int(f_a,-1,1,tol)
    
    plotting_init(f_a, int_a, -1, 1, 20)
    plotting_tol(f_a, -1, 1, -5)

    #print (gaussq_tol(math.sin, 0, math.pi , 0.0001))
    #print (gaussq_n(math.sin, 0, math.pi , gaussq_tol(math.sin, 0, math.pi , 0.0001)))
    test_int(math.sin, 0, math.pi , tol)

    #print (gaussq_tol(f_c, -2, 3, 0.0001))
    #print (gaussq_n(f_c, -2, 3 , gaussq_tol(f_c, -2, 3, 0.0001)))
    test_int(f_c,-2,3, tol)
main()