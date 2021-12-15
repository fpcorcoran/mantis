import numpy as np
import numpy.linalg as LA

def estimate_noise(Y,R_hat):
    """Estimate the signal and noise corr. matrices"""
    R_ = LA.pinv(R_hat)

    #dimensionality (# of bands)
    D = R_.shape[0]
    N = Y.shape[1]

    Xi = []
    for i in range(D):
        #get all indices EXCEPT i
        di = np.arange(D)!=i

        #row i of R_ with column i removed
        R_di_i = R_[di,i].reshape(-1,1)
        R_i_di = R_[i,di].reshape(1,-1)

        #column i of R_ with row i removed
        R_hat_di_i = R_hat[di,i].reshape(-1,1)

        #R_ with both row i, column i removed
        R_di_di = R_[di][:,di]

        #regression vector
        Bi = (R_di_di - np.dot(R_di_i, R_i_di) / R_[i,i]) @ R_hat_di_i

        #modeling noise vector
        xi = Y.T[:,i].reshape(-1,1) - (Y.T[:,di] @ Bi).reshape(-1,1)

        Xi.append(xi)

    Xi = np.array(Xi).reshape(D,-1)

    #noise correlation estimation
    Rn = np.dot(Xi,Xi.T) / N

    #signal correlation estimation
    Rx = np.dot((Y - Xi),(Y - Xi).T) / N

    return Rn, Rx

def diracs(E,Rn,Ry):
    """Evaluate the diracs of the objective function."""
    d = []

    #loop through eigvecs
    for e in E.T:
        #signal power term
        p = e.T@Ry@e

        #noise power term
        sig = e.T@Rn@e

        #objective function
        d.append((2*sig)-p)

    return np.array(d)

def HySiME(Y, developing=False):
    """
    Hyperspectral Signal Identification
    by Minimum Error:

    Used to determine the signal subspace
    from a hyperspectral image and estimate
    the number of endmembers present in the
    image.

    Author
    ------
    Forrest Corcoran
    fpcorcoran17@gmail.com
    
    Source
    ------
    Hyperspectral Subspace Identification, 2008
    J.M. Bioucas-Dias, J.M.P. Nascimento
    10.1109/tgrs.2008.918089IEEE
    Transactions on Geoscience and Remote Sensing

    Parameters
    ----------
    Y (ndarray):
        Array containing the image spectra.
        dims = (# bands, # pixels)

    Returns
    -------
    ndarray
        The hyperspectral signal subspace.
        dims = (# bands, # endmembers)

    """

    #number of pixels
    N = Y.shape[1]

    #sample correlation matrix
    R_hat = np.dot(Y,Y.T)

    #sample correlation matrix
    Ry = R_hat / N

    #estimate signal, noise correlation matrices
    Rn, Rx = estimate_noise(Y, R_hat)

    #signal correlation matrix eigenvectors
    E = LA.eig(Rx)[1]

    #calculate diracs from objective function
    d = diracs(E,Rn,Ry)

    #get indices of negative diracs
    i = d<0

    #get eigvecs with negative diracs (i.e. signal subspace)
    Us = E.T[i]

    if developing:
        return Us.T, E, Rn, Rx, Ry
    else:
        return Us.T
