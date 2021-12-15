import numpy as np
import numpy.linalg as LA

def RMUSIC(Y, D, Us, alpha=0.85):
    """
    Robust MUltiple SIgnal Classification:

    Used to pre-select a number of spectra
    from a larger spectral dictionary based
    on the hyperspectral image signal
    subspace.

    Source
    ------
    Semiblind Hyperspectral Unmixing in
    the Presence of Spectral Library
    Mismatches, 2016
    Xiao Fu, Wing-Kin Ma,
    Jose M. Bioucas-Dias, Tsung-Han Chan
    10.1109/tgrs.2016.2557340IEEE
    Transactions on Geoscience and Remote Sensing

    Parameters
    ----------
    Y (ndarray):
        Array containing the image spectra.
        dims = (# bands, # pixels)
    D (ndarray):
        Array containing the library spectra.
        dims = (# bands, # spectra)
    Us (ndarray):
        The signal subspace. Can be determined
        via HySiME or other method (SVD, etc.).
    alpha (float): optional
        Hyperparameter governing allowable
        error threshold between library
        and image spectra.

    Returns
    -------
    ndarray
        Pruned dictionary containing RMUSIC
        metric for each entry in the
        spectral dictionary. Smaller values
        indicate a better match.

    """

    #Calculate error factor (epsilon)
    dk_min = np.min(np.apply_along_axis(LA.norm,1,D))
    e = ((1-alpha)/(1+alpha)) * dk_min

    #signal subspace projector and orthogonal complement
    P_Us = np.dot(Us,Us.T)
    P_Us_ort = np.identity(P_Us.shape[0]) - P_Us

    #test range of theta < e (error term)
    theta = np.linspace(0,e)

    nu_star = []
    for k in range(D.shape[1]):
        #Get dictionary entry # k
        dk = D[:,k]

        nu = []
        for t in theta:
            #numerator and denominator of nu
            num = np.abs(LA.norm(np.dot(P_Us_ort,dk)) - t)
            den = LA.norm(np.dot(P_Us,dk)) + np.sqrt(e**2 - t**2)

            nu.append(num/den)

        #nu* = min(nu)
        nu_star.append(np.min(nu))

    nu_star = np.asarray(nu_star)

    #compute robust MUSIC spectra
    return (nu_star**2)/(nu_star**2 + 1)
