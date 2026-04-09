# Yash Goyal 24110399 Tutorial 1 Ques4: 

import math

def bisection(f, a, b, tol=1e-4):
    
    if f(a) * f(b) >= 0:
        return None
    
    while (b - a)/2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2


def f1(x):
    return math.sqrt(x) - math.cos(x)

def f2(x):
    return math.sin(x) + 2

def f3(x):
    return x * math.exp(x)


print("Root for (i):", bisection(f1,0,1))  #Result: Root for (i): 0.64166259765625
print("Root for (ii):", bisection(f2,2,6)) #Result: Root for (ii): None
print("Root for (iii):", bisection(f3,-2,0.5)) #Result: Root for (iii): 4.57763671875e-05

print('''
(a) Bisection works only when the function changes sign in the interval.

(i) Works because sign change exists.

(ii) Does not work because sin(x)+2>0 (positive)always.

(iii) Works because sign change occurs.

(b) Absolute error is more appropriate in 3rd part because the root is near 0. Relative error divides the difference between approximations by the value of the root, and when this value becomes very small (close to 0), the denominator becomes extremely small or even zero. This can make the relative error very large or undefined, even when the approximation is actually very close to the true root.
''')