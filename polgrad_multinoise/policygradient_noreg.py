import numpy as np
from numpy import linalg as la
from matrixmath import solveb
from printing import inplace_print

from time import time,sleep
import warnings
from warnings import warn
from copy import copy


def run_policy_gradient(SS,PGO,CSO=None):

    # Initialize
    stop = False
    converged = False
    stop_early = False
    iterc = 0
    t_start = time()
    headerstr = 'Iteration | Stop quant / threshold |  Curr obj |  Best obj | Norm of gain delta | Stepsize '
    print(headerstr)

    K0 = SS.K
    K = SS.K
    Kbest = SS.K
    objfun_best = np.inf
    Kold = K0

    P = SS.P
    S = SS.S

    # Initialize history matrices
    if PGO.keep_hist:
        mat_shape = list(K.shape)
        mat_shape.append(PGO.max_iters)
        mat_shape = tuple(mat_shape)
        K_hist = np.full(mat_shape, np.inf)
        grad_hist = np.full(mat_shape, np.inf)
        c_hist = np.full(PGO.max_iters, np.inf)
        objfun_hist = np.full(PGO.max_iters, np.inf)


    # Iterate
    while not stop:

        if PGO.exact:
            # Calculate gradient (G)
            # Do this to get combined calculation of P and S,
            # pass previous P and S to warm-start dlyap iterative algorithm
            SS.calc_PS(P,S)
    #        SS.calc_PS(P,S,'linsolve')
            Glqr = SS.grad
            P = SS.P
            S = SS.S

        # Calculate step direction (V)
        if PGO.regularizer is None:
            G = Glqr


        if PGO.regularizer is None:
            if PGO.step_direction=='gradient':
                V = G
            elif PGO.step_direction=='natural_gradient':
                V = solveb(SS.grad,SS.S)
            elif PGO.step_direction=='gauss_newton':
                V = solveb(la.solve(SS.RK,SS.grad),SS.S)

        # Check if mean-square stable
        if SS.c == np.inf:
            raise Exception('ITERATE WENT UNSTABLE DURING GRADIENT DESCENT')

        if PGO.regularizer is None:
            objfun = SS.c

        # Record current iterate
        if PGO.keep_hist:
            K_hist[:,:,iterc] = SS.K
            grad_hist[:,:,iterc] = SS.grad
            c_hist[iterc] = SS.c
            objfun_hist[iterc] = objfun

        if iterc == 0:
            Kchange = np.inf
        else:
            Kchange = la.norm(K-Kold,'fro')/la.norm(K,'fro')
        Kold = K

        # Check for stopping condition
        if PGO.stop_crit=='gradient':
            normgrad = la.norm(G)
            stop_quant = normgrad
            stop_thresh = PGO.epsilon
            if normgrad < PGO.epsilon:
                converged = True
        elif PGO.stop_crit=='Kchange':
            stop_quant = Kchange
            stop_thresh = PGO.epsilon
            if Kchange < PGO.epsilon:
                converged = True
        elif PGO.stop_crit=='fixed':
            stop_quant = iterc
            stop_thresh = PGO.max_iters

        if iterc >= PGO.max_iters-1:
            stop_early = True
        else:
            iterc += 1

        stop = converged or stop_early

        # Record current best (subgradient method)
        if objfun < objfun_best:
            objfun_best = objfun
            Kbest = SS.K
            fbest_repeats = 0
        else:
            fbest_repeats += 1

        # Update iterate
        if PGO.opt_method == 'gradient':
            if PGO.step_direction=='policy_iteration':
                eta = 0.5 # for printing only
                H21 = la.multi_dot([SS.B.T,SS.P,SS.A])
                H22 = SS.RK
                if PGO.regularizer is None:
                    K = -la.solve(H22,H21)
                    SS.setK(K)
            else:
                # Calculate step size
                if PGO.stepsize_method=='constant':
                    eta = PGO.eta
                    K = SS.K - eta*V
                    SS.setK(K)

        if hasattr(PGO,'slow'):
            if PGO.slow:
                sleep(0.05)

        # Printing
        if PGO.display_output:
            # Print iterate messages
            printstr0 = "{0:9d}".format(iterc)
            printstr1 = " {0:5.3e} / {1:5.3e}".format(stop_quant, stop_thresh)
            printstr2a = "{0:5.3e}".format(objfun)
            printstr2b = "{0:5.3e}".format(objfun_best)
            printstr3 = "         {0:5.3e}".format(Kchange)
            printstr4 = "{0:5.3e}".format(eta)
            printstr = printstr0+' | '+printstr1+' | '+printstr2a+' | '+printstr2b+' | '+printstr3+' | '+printstr4+' |'
            if PGO.display_inplace:
                if iterc==0:
                    print(" " * len(printstr),end='')
                inplace_print(printstr)
            else:
                print(printstr)
            if stop: # Print stopping messages
                print('')
                if converged:
                    print('Optimization converged, stopping now')
                if stop_early:
#                    warnings.simplefilter('always', UserWarning)
#                    warn('Max iterations exceeded, stopping optimization early')
                    print('Max iterations exceeded, stopping optimization')

    if PGO.keep_hist:
        # Trim empty parts from preallocation
        K_hist = K_hist[:,:,0:iterc+1]
        grad_hist = grad_hist[:,:,0:iterc+1]
        c_hist = c_hist[0:iterc+1]
        objfun_hist = objfun_hist[0:iterc+1]
    else:
        K_hist = None
        grad_hist = None
        c_hist = None
        objfun_hist = None

    if PGO.keep_opt == 'best':
        SS.setK(Kbest)

    t_end = time()

    hist_list = [K_hist,grad_hist,c_hist,objfun_hist]

    print('Policy gradient descent optimization completed after %d iterations, %.3f seconds' % (iterc,t_end-t_start))
    return SS,hist_list