import numpy as np
import numpy.random as npr
import numpy.linalg as la
from matrixmath import specrad, mdot, sympart, positive_semidefinite_part

# Vectorize a matrix by stacking its columns
def vec(A):
    return A.reshape(-1, order="F")

def groupdot(A,x):
    return np.einsum('...jk,...k',A,x)


def reshaper(X,m,n,p,q):
    Y = np.zeros([m*n,p*q])
    k = 0
    for j in range(n):
        for i in range(m):
            Y[k] = vec(X[i*p:(i+1)*p,j*q:(j+1)*q])
            k += 1
    return Y


def prettyprint(A,matname=None,fmt='%+13.9f'):
    print("%s = " % matname)
    if len(A.shape)==2:
        n = A.shape[0]
        m = A.shape[1]
        for icount,i in enumerate(A):
            print('[' if icount==0 else ' ', end='')
            print('[',end='')
            for jcount,j in enumerate(i):
                print(fmt % j,end=' ')
            print(']', end='')
            print(']' if icount==n-1 else '', end='')
            print('')


def ctrb(A, B):
    n,m = np.shape(B)
    CTRB = np.zeros([n,n*m])

    # Construct the controllability matrix
    Ai = np.eye(n)

    for i in range(n):
        cols = np.arange(i*m,(i+1)*m)
        CTRB[:,cols] = np.dot(Ai,B)
        Ai = np.dot(Ai,A)

    return CTRB


def generate_sample_data(n,m,SigmaA,SigmaB,nr,ell,u_mean_std=1.0,u_covr_var=0.1):
    # Generate the random input means and covariances
    u_mean_hist = np.zeros([ell, m])
    u_covr_hist = np.zeros([ell, m, m])
    for t in range(ell):
        # Sample the means from a Gaussian distribution
        u_mean_hist[t] = u_mean_std*npr.randn(m)
        # Sample the covariances from a Wishart distribution
        u_covr_base = u_covr_var*npr.randn(m,m)
        u_covr_hist[t] = np.dot(u_covr_base.T,u_covr_base)

    # Generate the inputs
    u_hist = np.zeros([ell, nr, m])
    for t in range(ell):
        u_mean = u_mean_hist[t]
        u_covr = u_covr_hist[t]
        u_hist[t] = npr.multivariate_normal(u_mean, u_covr, nr)

    # Generate the process noise
    Anoise_vec_hist = npr.multivariate_normal(np.zeros(n * n), SigmaA, [ell, nr])
    Bnoise_vec_hist = npr.multivariate_normal(np.zeros(n * m), SigmaB, [ell, nr])
    Anoise_hist = np.reshape(Anoise_vec_hist, [ell, nr, n, n], order='F')
    Bnoise_hist = np.reshape(Bnoise_vec_hist, [ell, nr, n, m], order='F')

    return u_mean_hist,u_covr_hist,u_hist,Anoise_hist,Bnoise_hist


def collect_rollouts(n,m,A,B,nr,ell,Anoise_hist,Bnoise_hist,u_hist,print_updates=False):
    # Collect rollout data
    x_hist = np.zeros([ell+1, nr, n])

    # Initialize the state
    x_hist[0] = npr.randn(nr, n)

    # Transition the state
    for t in range(ell):
        x_hist[t+1] = groupdot(A+Anoise_hist[t], x_hist[t]) + groupdot(B+Bnoise_hist[t],u_hist[t])
        if print_updates:
            print("Simulated timestep %d / %d" % (t+1,ell))

    return x_hist


def estimate_model(n, m, A, B, SigmaA, SigmaB, nr, ell, x_hist, u_mean_hist, u_covr_hist,
                   display_estimates=False, AB_known=False):
    muhat_hist = np.zeros([ell+1, n])
    Xhat_hist = np.zeros([ell+1, n*n])
    What_hist = np.zeros([ell+1, n*m])

    # First stage: mean dynamics parameter estimation
    if AB_known:
        Ahat = np.copy(A)
        Bhat = np.copy(B)
    else:
        # Form data matrices for least-squares estimation
        for t in range(ell+1):
            muhat_hist[t] = (1/nr)*np.sum(x_hist[t], axis=0)
            Xhat_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j', x_hist[t], x_hist[t]), axis=0))
            if t < ell:
                # What_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j',x_hist[t],u_mean_hist[t]),axis=0))
                What_hist[t] = vec(np.outer(muhat_hist[t], u_mean_hist[t]))
        Y = muhat_hist[1:].T
        Z = np.vstack([muhat_hist[0:-1].T, u_mean_hist.T])
        # Solve least-squares problem
        # Thetahat = mdot(Y, Z.T, la.pinv(mdot(Z, Z.T)))
        Thetahat = la.lstsq(Z.T, Y.T, rcond=None)[0].T
        # Split learned model parameters
        Ahat = Thetahat[:,0:n]
        Bhat = Thetahat[:,n:n+m]

    AAhat = np.kron(Ahat, Ahat)
    ABhat = np.kron(Ahat, Bhat)
    BAhat = np.kron(Bhat, Ahat)
    BBhat = np.kron(Bhat, Bhat)

    # Second stage: covariance dynamics parameter estimation
    # Form data matrices for least-squares estimation
    C = np.zeros([ell, n*n]).T
    Uhat_hist = np.zeros([ell, m*m])
    for t in range(ell):
        Uhat_hist[t] = vec(u_covr_hist[t] + np.outer(u_mean_hist[t], u_mean_hist[t]))
        Cminus = mdot(AAhat,Xhat_hist[t])+mdot(BAhat,What_hist[t])+mdot(ABhat,What_hist[t].T)+mdot(BBhat,Uhat_hist[t])
        C[:,t] = Xhat_hist[t+1] - Cminus
    D = np.vstack([Xhat_hist[0:-1].T, Uhat_hist.T])
    # Solve least-squares problem
    # SigmaThetahat_prime = mdot(C, D.T, la.pinv(mdot(D,D.T)))
    SigmaThetahat_prime = la.lstsq(D.T, C.T, rcond=None)[0].T
    # Split learned model parameters
    SigmaAhat_prime = SigmaThetahat_prime[:, 0:n*n]
    SigmaBhat_prime = SigmaThetahat_prime[:, n*n:n*(n+m)]

    # Reshape and project the noise covariance estimates onto the semidefinite cone
    SigmaAhat = reshaper(SigmaAhat_prime, n, n, n, n)
    SigmaBhat = reshaper(SigmaBhat_prime, n, m, n, m)
    SigmaAhat = positive_semidefinite_part(SigmaAhat)
    SigmaBhat = positive_semidefinite_part(SigmaBhat)

    if display_estimates:
        prettyprint(Ahat, "Ahat")
        prettyprint(A, "A   ")
        prettyprint(Bhat, "Bhat")
        prettyprint(B, "B   ")
        prettyprint(SigmaAhat, "SigmaAhat")
        prettyprint(SigmaA, "SigmaA   ")
        prettyprint(SigmaBhat, "SigmaBhat")
        prettyprint(SigmaB, "SigmaB   ")

    return Ahat, Bhat, SigmaAhat, SigmaBhat


def estimate_model_var_only(n, m, A, B, SigmaA, SigmaB, varAi, varBj, Ai, Bj, nr, ell, x_hist, u_mean_hist, u_covr_hist,
                            display_estimates=False, AB_known=False, detailed_outputs=True):
    muhat_hist = np.zeros([ell+1, n])
    Xhat_hist = np.zeros([ell+1, n*n])
    What_hist = np.zeros([ell+1, n*m])

    # First stage: mean dynamics parameter estimation
    if AB_known:
        Ahat = np.copy(A)
        Bhat = np.copy(B)
    else:
        # Form data matrices for least-squares estimation
        for t in range(ell+1):
            muhat_hist[t] = (1/nr)*np.sum(x_hist[t], axis=0)
            Xhat_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j', x_hist[t], x_hist[t]), axis=0))
            if t < ell:
                # What_hist[t] = (1/nr)*vec(np.sum(np.einsum('...i,...j',x_hist[t],u_mean_hist[t]),axis=0))
                What_hist[t] = vec(np.outer(muhat_hist[t], u_mean_hist[t]))
        Y = muhat_hist[1:].T
        Z = np.vstack([muhat_hist[0:-1].T, u_mean_hist.T])
        # Solve least-squares problem
        # Thetahat = mdot(Y, Z.T, la.pinv(mdot(Z, Z.T)))
        Thetahat = la.lstsq(Z.T, Y.T, rcond=None)[0].T
        # Split learned model parameters
        Ahat = Thetahat[:,0:n]
        Bhat = Thetahat[:,n:n+m]

    AAhat = np.kron(Ahat, Ahat)
    ABhat = np.kron(Ahat, Bhat)
    BAhat = np.kron(Bhat, Ahat)
    BBhat = np.kron(Bhat, Bhat)

    # Second stage: covariance dynamics parameter estimation
    # Form data matrices for least-squares estimation
    C = np.zeros([ell, n*n]).T
    Uhat_hist = np.zeros([ell, m*m])
    for t in range(ell):
        Uhat_hist[t] = vec(u_covr_hist[t] + np.outer(u_mean_hist[t], u_mean_hist[t]))
        Cminus = mdot(AAhat,Xhat_hist[t])+mdot(BAhat,What_hist[t])+mdot(ABhat,What_hist[t].T)+mdot(BBhat,Uhat_hist[t])
        C[:,t] = Xhat_hist[t+1] - Cminus

    C = vec(C)

    X1 = Xhat_hist[0:-1].T
    U1 = Uhat_hist.T

    D1 = np.vstack([vec(np.dot(np.kron(Ai[i], Ai[i]), X1)) for i in range(np.size(varAi))])
    D2 = np.vstack([vec(np.dot(np.kron(Bj[j], Bj[j]), U1)) for j in range(np.size(varBj))])

    D = np.vstack([D1, D2])

    # Solve least-squares problem
    # var_hat = mdot(C, D.T, la.pinv(mdot(D,D.T)))
    # var_hat = mdot(la.pinv(mdot(D, D.T)), D, C)
    var_hat = la.lstsq(D.T, C, rcond=None)[0]

    varAi_hat = np.maximum(var_hat[0:np.size(varAi)], 0)
    varBj_hat = np.maximum(var_hat[np.size(varAi):], 0)

    SigmaAhat = np.sum([varAi_hat[i]*np.outer(vec(Ai[i]), vec(Ai[i])) for i in range(np.size(varAi))], axis=0)
    SigmaBhat = np.sum([varBj_hat[j]*np.outer(vec(Bj[j]), vec(Bj[j])) for j in range(np.size(varBj))], axis=0)

    if display_estimates:
        prettyprint(Ahat, "Ahat")
        prettyprint(A, "A   ")
        prettyprint(Bhat, "Bhat")
        prettyprint(B, "B   ")
        prettyprint(SigmaAhat, "SigmaAhat")
        prettyprint(SigmaA, "SigmaA   ")
        prettyprint(SigmaBhat, "SigmaBhat")
        prettyprint(SigmaB, "SigmaB   ")

    if detailed_outputs:
        outputs = Ahat, Bhat, SigmaAhat, SigmaBhat, varAi_hat, varBj_hat
    else:
        outputs = Ahat, Bhat, SigmaAhat, SigmaBhat
    return outputs