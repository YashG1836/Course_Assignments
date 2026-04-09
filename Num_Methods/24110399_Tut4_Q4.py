# Yash Goyal 24110399 
# Tut 4 Ques 4

import numpy as np

def f(x):
    return x**2 * np.exp(x)
def function(a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(1, n):
        x = a + i*h
        sum += f(x)
    result = (h/2) * (f(a) + f(b) + 2*sum)
    return result


exact = 13.4226
print("Given exact value =", round(exact,2))

arr = [4, 8, 16]

for n in arr:
    approx = function(-2, 2, n)
    error = abs(approx - exact)
    print("")
    print("For n =", n)
    print("Value =", round(approx, 2))
    print("Error =", round(error, 2))
    
'''
Results: 
Given exact value = 13.42

For n = 4
Value = 18.13
Error = 4.71

For n = 8
Value = 14.64
Error = 1.22

For n = 16
Value = 13.73
Error = 0.31

'''
