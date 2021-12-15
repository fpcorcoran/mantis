import numpy as np
from numpy.linalg import norm, inv
from time import time
from copy import deepcopy

def soft_shrinkage(X,tao):
    """soft shrinkage thresholding operator"""
    
    mask = np.abs(X) - tao <= 0
    
    X[mask] = 0
    
    return X

def compute_C(A1):
    """compute the constraint matrix"""
    C = []
    for i in range(A1.shape[0]):
        for j in range(A1.shape[0]):
            if i != j:
                C.append(A1[i,:]*A1[j,:])
                
    C = np.unique(np.array(C),axis=0)

    return C

def NU_BGBM(Y,E,F,W,_lambda,mu,epsilon,max_iter):
    """
    Nonlinear Unmixing of Bandwise 
    Generalized Bilinear Model:

    ADMM based solution to solving
    the Generalized Bilinear Mixing
    Model (GBMM). Takes into account
    both dense and sparse noise on 
    bandwise basis.

    Author
    ------
    Forrest Corcoran
    fpcorcoran17@gmail.com
    
    Source
    ------
    Li, C., Liu, Y., Cheng, J., 
    Song, R., Peng, H., Chen, Q., 
    & Chen, X. (2018). 
    Hyperspectral Unmixing with
    Bandwise Generalized Bilinear Model. 
    Remote Sensing, 10(10), 1600.
    https://doi.org/10.3390/rs10101600

    Parameters
    ----------
    Y (ndarray):
        Array containing the image spectra.
        dims = (# bands, # pixels)
        
    E (ndarray):
        Array containing the endmember 
        spectral library.
        dims = (# bands, # endmembers)
        
    F (ndarray):
        Array containing mixed endmember
        spectral library.
        dims = (# bands, # mixed endmembers)
        
    W (ndarray):
        Diagonal matrix. W_{i,i} contains 
        variance of spectral band i.
        dims = (# bands, # bands)
        

    Returns
    -------
    A1 (ndarray):
        Endmember abundance matrix.
        dims = (# endmembers, # pixels)
        
    B1 (ndarray):
        Mixed endmember abundance matrix.
        dims = (# mixed endmembers, # pixels)
        
    S1 (ndarray):
        Sparse noise image containing artifacts
        in each band.
        dims = (# bands, # pixels)

    """
    
    #Get dimensions
    M, D, P = E.shape[1], Y.shape[0], Y.shape[1]
    
    #initialize A,B,S matrices
    A0 = np.abs(np.random.normal(0,1,size=(E.shape[1],Y.shape[1])))
    B0 = np.abs(np.random.normal(0,1,size=(F.shape[1],Y.shape[1])))
    S0 = np.abs(np.random.normal(0,1,size=Y.shape))
    
    A1 = np.abs(np.random.normal(0,1,size=(E.shape[1],Y.shape[1])))
    B1 = np.abs(np.random.normal(0,1,size=(F.shape[1],Y.shape[1])))
    S1 = np.abs(np.random.normal(0,1,size=Y.shape))
    
    C = compute_C(A1)
        
    V1 = np.abs(np.random.normal(0,1,size=S1.shape))
    V2 = np.abs(np.random.normal(0,1,size=A1.shape))
    V3 = np.abs(np.random.normal(0,1,size=B1.shape))
    
    L1 = np.abs(np.random.normal(0,1,size=S1.shape))
    L2 = np.abs(np.random.normal(0,1,size=A1.shape))
    L3 = np.abs(np.random.normal(0,1,size=B1.shape))
    
    I_E = np.identity(E.shape[1])
    I_F = np.identity(F.shape[1])
    I_V = np.identity(W.shape[0])

    
    #stopping criteria divisor
    n = np.sqrt((3*(M+D))/P)
    
    #primal residual
    r = epsilon+1
    
    #dual residual
    d = epsilon+1
    
    #store primal and dual residuals
    primal, dual = [], []
    
    k=1
    t0 = time()
    while (r/n > epsilon) and (d/n > epsilon) and (k < max_iter):
        print(k)
        
        #ADMM Update Steps
        
        #Primal Variables
        A1 = inv((W@E).T@(W@E)+(mu*I_E)) @ ((W@E).T@W@(Y - F@B1 - V1) + mu*(V2-L2))
        B1 = inv((W@F).T@(W@F)+(mu*I_F)) @ ((W@F).T@W@(Y - E@A1 - V1) + mu*(V3-L3))
        
        S1 = soft_shrinkage(V1-L1, _lambda/mu)
        
        #Dual Variables
        V1 = inv(W.T@W+(mu*I_V)) @ (W.T@W@(Y - E@A1 - F@B1) + mu*(S1-L1))
        V2 = np.maximum(A1+L2, np.zeros_like(A1+L2))
        V3 = np.minimum(np.maximum(B1+L3, np.zeros_like(B1+L3)), C)
        
        #Multipliers
        L1 = L1 - (V1-S1) 
        L2 = L2 - (V2-A1)
        L3 = L3 - (V3-B1)
        
        #update residuals
        r = norm(S1-V1) + norm(A1-V2) + norm(B1-V3)
        d = mu * (norm(S1-S0) + norm(A1 - A0) + norm(B1 - B0))
        
        #store residuals
        primal.append(r)
        dual.append(-d)
        
        #update k to k+1
        S0 = deepcopy(S1)
        A0 = deepcopy(A1)
        B0 = deepcopy(B1)
        
        #update constraint matrix
        C = compute_C(A1)
        
        k+=1
        
    #enforce sum-to-one-constraint
    A1 /= np.apply_along_axis(np.sum,0,A1)
    B1 /= np.apply_along_axis(np.sum,0,B1)

        
    return A1,B1,S1,primal,dual