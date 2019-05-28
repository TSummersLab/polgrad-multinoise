# polgrad-multinoise

## Policy gradient methods for linear systems with multiplicative noise

The code in this repository implements the algorithms and ideas from our two papers:
* paper1 https://
* paper2 https://


## Dependencies
* Python 3.5+ (tested with 3.7.3)
* NumPy
* SciPy
* Matplotlib
* Control (requires Slycot)

## Installing


## Examples
There are several main Python files which can be run. 

1. main_model_known_example_suspension.py

This file reflects the model-known algorithms which calculate gradients
by solving generalized Lyapunov equations. The standard gradient is used with an
appropriate step size and the optimization terminates when the Frobenius norm
of the gradient falls below a threshold. The system represented is a 4-state,
1-input system with multiplicative noise as described in the paper. Both high
and zero multiplicative noise settings are optimized and the iterates recorded.
Running the routine_gen() function will perform the entire experiment,
while running the routine_load() function will attempt to load existing
experiment result data and generate the plots used in Figure 1 of the main paper.


2. main_model_known_example_random.py

This file reflects the model-known algorithms which calculate gradients
by solving generalized Lyapunov equations. The standard gradient, natural 
gradient, and Gauss-Newton step directions are used with appropriate step sizes
and fixed number of iterations.
Running the routine_gen() function will perform the entire experiment,
while running the routine_load() function will attempt to load existing
experiment result data and generate the plots used in Figure 2 of the main paper.


3. main_model_free_example.py

This file reflects the model-free algorithm which estimates gradients via 
zeroth-order optimization. This code is meant to simply give an idea how the 
algorithm works - the number of sample trajectories needed to estimate the 
gradient to the accuracy limit given in the paper is quite high and might take
a very long time to run, even for a small system. This code does not correspond
to any figure or numerical result given in the main paper.

4. main_sparse.py

This file reflects the sparse gain design procedures developed in []. Running this script will perform a "sparsity traversal" which iteratively increases the regularization weight in order to find increasingly sparse optimal gains.


## General code structure
Linear time-invariant (LTI) systems are represented by a simple class. For solving linear-quadratic regulator (LQR) optimal control problems additional attributes and methods are provided. Likewise, multiplicative noise is handled by additional attributes and methods. These class definitions are in ltimult.py. Random systems can be generated by the functions in ltimultgen.py. Policy gradient can be performed using policygradient.py.


## Authors
* **Ben Gravell** - [UTDallas](http://www.utdallas.edu/~tyler.summers/)
* **Yi Guo** - [UTDallas](http://www.utdallas.edu/~tyler.summers/)
* **Peyman Esfahani** - [TUDelft](http://www.dcsc.tudelft.nl/~mohajerin/)
* **Tyler Summers** - [UTDallas](http://www.utdallas.edu/~tyler.summers/)
