import numpy as np
from numpy import linalg as la
from scipy.linalg import solve_discrete_lyapunov,solve_discrete_are
#import control
from copy import copy
from functools import reduce

###############################################################################
# General matrix math functions
###############################################################################

# Check for positive definiteness
def is_pos_def(A):
    try:
        la.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Vectorize a matrix by stacking its columns
def vec(A):
    return A.reshape((-1, 1), order="F")

# Return the symmetric part of a matrix
def sympart(A):
    return 0.5*(A+A.T)

# Overload and extend the numpy kron function to take a single argument
def kron(*args):
    if len(args)==1:
        return np.kron(args[0],args[0])
    else:
        return np.kron(*args)

## Multi-dot
#def mdot(*args):
#    return la.multi_dot(args)

def mdot(*args):
    return reduce(np.dot, args)


# Similar to MATLAB / operator for square invertible matrices
# Solves a = bx
def solveb(a,b):
    return la.solve(b.T,a.T).T


# Overload the numpy randn function so it always uses the same RandomState
# similarly to how MATLAB works (random number generator is 'global')
seed = 3187
rng = np.random.RandomState(seed)

def rand(*args):
    return rng.rand(*args)

def randn(*args):
    return rng.randn(*args)

def randint(*args):
    return rng.randint(*args)

def rngg():
    return rng


# Symmetric log transform
def symlog(X,scale=1):
    return np.multiply(np.sign(X),np.log(1+np.abs(X)/(10**scale)))

# Ammend the dlyap and dare functions to correct issue where
# input A, Q matrices are modified (unwanted behavior);
# simply pass a copy of the matrices to protect them from modification
def dlyap(A,Q):
    try:
        return solve_discrete_lyapunov(copy(A),copy(Q))
    except ValueError:
        return np.full_like(Q,np.inf)

def dare(A,B,Q,R):
    return solve_discrete_are(copy(A),copy(B),copy(Q),copy(R))