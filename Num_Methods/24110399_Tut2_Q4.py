# Yash Goyal 24110399 Tutorial 2 Ques4: 

import math
TOL = 1e-6
MAX_ITER = 100

p0 = 1

def method_a(p):
    return (20/21)*p + 1/(p**2)

def method_b(p):
    return p - (p**3 - 21)/(3*p**2)

def method_c(p):
    return p - (p**4 - 21*p)/(p**2 - 21)

def method_d(p):
    return math.sqrt(21/p)

def fixed_point(g, name):
    p = p0
    for i in range(1, MAX_ITER + 1):
        try:
            p_new = g(p)
        except:
            print(f"{name}: Error (division by zero or invalid)")
            return None
        
        if abs(p_new - p) < TOL:
            print(f"{name}: Converged")
            print(f"  Root ≈ {p_new}")
            print(f"  Iterations = {i}\n")
            return i
        
        p = p_new

    print(f"{name}: Did NOT converge in {MAX_ITER} iterations\n")
    return None

# Run all methods
iters = {}

iters['a'] = fixed_point(method_a, "Method (a)")
iters['b'] = fixed_point(method_b, "Method (b)")
iters['c'] = fixed_point(method_c, "Method (c)")
iters['d'] = fixed_point(method_d, "Method (d)")

# Ranking convergence speed
print("=== Convergence Ranking ===")
valid = {k: v for k, v in iters.items() if v is not None}

sorted_methods = sorted(valid.items(), key=lambda x: x[1])

for method, it in sorted_methods:
    print(f"Method ({method}) → {it} iterations")
    
    
    
    
# Results 
''' 
Method (a): Converged
  Root ≈ 2.758918848324031
  Iterations = 76

Method (b): Converged
  Root ≈ 2.7589241763811208
  Iterations = 8

Method (c): Converged
  Root ≈ 0.0
  Iterations = 2

Method (d): Converged
  Root ≈ 2.7589240094959373
  Iterations = 24

=== Convergence Ranking ===
Method (c) → 2 iterations
Method (b) → 8 iterations
Method (d) → 24 iterations
Method (a) → 76 iterations'''