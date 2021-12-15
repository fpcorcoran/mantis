import numpy as np
import numpy.linalg as LA

def DANSER(Y, D_tilda, stop_value=0.1, max_iter=200, alpha=0.85,_lambda=0.5, tao=10e-6, p=0.5, mu=10e5):
    """
    Dictionary Adjusted Non-convex
    Sparsity Encouraged Regression:

    Adaptation of collaborative
    sparse regression algorithm
    that is better equipped to handle
    small mismatches between image and
    library spectra. Used to generate
    abundance maps from the AOI image.

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
    D_tilda (ndarray):
        Array containing the (pruned)
        library spectra.
        dims = (# bands, # spectra)
    stop_value (float): optional
        The halting value for the regression.
        When the Frobenius norm of the
        difference between abundances on
        iteration i, i+1 is less than
        stop_value, the algorithm halts.
    max_iter (int): optional
        The maximum number of iterations
        before halting.
    alpha (float): optional
        Hyperparameter governing allowable
        error threshold between library
        and image spectra.
    _lambda (float): optional
        Hyperparameter used in updating the
        slack variable of the objective
        function.
    tao (float): optional
        Hyperparameter used in updating the
        slack variable of the objective
        function.
    p (float): optional
        Hyperparameter used in updating the
        slack variable of the objective
        function.
    mu (float): optional
        Hyperparameter governing allowable
        magnitude of spectral adjustment.
        A smaller mu allows for larger
        adjustment, and vice versa.


    Returns
    -------
    ndarray
        Result of the collaborative
        sparse regression.
        dims = (# endmembers, # pixels)

    """

    #D_tilda is mutable, so we must copy it
    _D_tilda = D_tilda.copy()

    #Calculate error factor (epsilon)
    dk_min = np.min(np.apply_along_axis(LA.norm,1,_D_tilda))
    e = ((1-alpha)/(1+alpha)) * dk_min

    #initialize endmember abundance matrix
    C = np.random.normal(0,1,(_D_tilda.shape[1],Y.shape[1]))

    #Steps for iterative optimization
    wk = [(p/2)*(LA.norm(C[k,:])**2 + tao) for k in range(C.shape[0])]
    wk = np.asarray(wk)

    #Initialize slack variable H matrix
    I = np.identity((C@C.T).shape[0])
    H = np.dot((mu*_D_tilda + np.dot(Y,C.T)), LA.inv(C@C.T + mu*I))

    #define stopping criteria
    stop=stop_value

    #iteration number
    i=0

    #initialize convergence metric
    convergence = LA.norm(Y - _D_tilda@C)
    progress = []
    #until stopping criteria is met
    while (convergence > stop):
        
        #define theta for slack variable update
        theta = np.sqrt(wk*_lambda)

        #update slack variable
        H_tilda = np.concatenate((np.sqrt(1/2)*H, np.diag(theta)), axis=0)

        #save shape
        s = (len(theta),Y.shape[1])

        #update Y
        Y_tilda = np.concatenate((np.sqrt(1/2)*Y,np.zeros(s)),axis=0)

        F = np.dot(Y_tilda.T,H_tilda)
        G = np.dot(H_tilda.T,H_tilda)

        #iterate over abundance maps
        for k in range(C.shape[0]):
            #
            ck = (F[:,k]-C.T@G[:,k] + C[k,:].T*G[k,k]) / G[k,k]

            #enforce non-negativity constraint
            idx = ck < 0
            ck[idx] = 0

            C[k,:] = ck

        for k in range(C.shape[0]):
            dk = _D_tilda[:,k]
            hk = H[:,k]

            #check if hk - dk distance within threshold
            if LA.norm(hk-dk) <= e:
                #update endmember with h
                _D_tilda[:,k] = hk

            else:
                #adjust endmember in direction of hk
                _D_tilda[:,k] = dk + e*((hk-dk)/LA.norm(hk-dk))

        #update slack variable with adjusted endmembers
        I = np.identity((C@C.T).shape[0])
        H = np.dot((mu*_D_tilda + np.dot(Y,C.T)), LA.inv(C@C.T + mu*I))

        wk = [(p/2)*(LA.norm(C[k,:])**2 + tao) for k in range(C.shape[0])]
        wk = np.array(wk)

        #update convergence metric
        convergence = np.abs(convergence - LA.norm(Y - _D_tilda@C))
        progress.append(LA.norm(Y - D_tilda@C))
        #break the loop if max iterations are reached
        if i == max_iter:
            print("Max Iters Reached")
            break

        i+=1

    #enforce sum-to-one-constraint
    C /= np.apply_along_axis(np.sum,0,C)

    return C, _D_tilda, progress
