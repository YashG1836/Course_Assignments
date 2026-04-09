# Name: Yash Goyal , 24110399
# Tut 3 Ques 4
import numpy as np

def f1(x):
    return np.cos(x)
def f2(x):
    return 1 / (1 + 25 * x**2)

def lagrange(x_p, y_p, x):
    n = len(x_p)
    result = 0
    for i in range(n):
        term = y_p[i]
        for j in range(n):
            if i != j:
                term *= (x-x_p[j])/(x_p[i]-x_p[j])
        result += term
    return result


arr = [5, 25, 50]
test = [0, 0.95]

for n in arr:
    print("")
    print("For N =", n, ":")
    print("")
    
    x_p = np.linspace(-1, 1, n)
    
    y1 = f1(x_p)
    y2 = f2(x_p)
    
    for x in test:
        p1 = lagrange(x_p, y1, x)
        exact1 = f1(x)
        error1 = abs(p1 - exact1)
        
        p2 = lagrange(x_p, y2, x)
        exact2 = f2(x)
        error2 = abs(p2 - exact2)
        
        print(f"x = {x}")
        print(f"f1 error = {error1:.2f}")
        print(f"f2 error = {error2:.2f}")
        
        
        
# The Results of the above code :
'''
For N = 5 :

x = 0
f1 error = 0.00
f2 error = 0.00
x = 0.95
f1 error = 0.00
f2 error = 0.20

For N = 25 :

x = 0
f1 error = 0.00
f2 error = 0.00
x = 0.95
f1 error = 0.00
f2 error = 115.04

For N = 50 :

x = 0
f1 error = 0.00
f2 error = 0.00
x = 0.95
f1 error = 0.00
f2 error = 17957.49
'''

'''
(C) For the function F1, the least error occurs at N=50 , considering cos(x), its a smooth function and incresing n would infact impore the approximations and the error decreses.
The other function F2, it ossiclates near the boundary values and hence incresing n would increse the error as there will be a lot of change in the approximated values. so the lowest error occurs for N=5.
'''
