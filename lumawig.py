import numpy as np

def bottleneck0(X,Y):
    """
    Returns the dimension 0 bottleneck distance between two non-empty persistence diagrams.
    All components are assumed to be born at the beginning of the filtration, 
    that is, all non-trivial points lie in the vertical axis.   
    
    Parameters
    ----------
    X : 1D numpy array
        Non-empty array of death times for persistence diagram 1.
    Y : 1D numpy array
        Non-empty array of death times for persistence diagram 2.
           
    Returns
    -------
    d : float
        The bottleneck distance between X and Y.

    Authors
    --------
    Paul Samuel P. Ignacio,
    University of the Philippines Baguio,
    Governor Pack Road, Baguio City 2600, Philippines
    
    Jay-Anne B. Bulauan,
    University of the Philippines Baguio,
    Governor Pack Road, Baguio City 2600, Philippines

    David Uminsky,
    University of Chicago, 5801 S Ellis Ave, Chicago,IL 60637, United States
        
    Examples
    --------
    >>> num_points1 = 100
    >>> num_points2 = 200
    >>> X = np.random.uniform(low=0.0, high=1000.0, size=num_points1)
    >>> Y = np.random.uniform(low=0.0, high=1000.0, size=num_points2)
    >>> bottleneck0(X,Y)
   
      
    """
    
    #Swap if X is longer
    if len(Y) < len(X):
        X_copy = np.copy(X)
        X = Y
        Y = X_copy  
    
    #Initialize
    X=-np.sort(-X) 
    Y=-np.sort(-Y)
    d=0
    N = len(X)

    Z = abs(X-Y[0:N]) 
    l = np.argmax(Z) 
    dtemp = Z[l]

    # Maintain an initial bijection. Systematically modify this bijection to optimize
    # the norm between matched points until the bottleneck matching is recovered.
    if N != len(Y) and dtemp < 0.5*Y[N]:
        d = 0.5*Y[N]      
    elif l >= 0 and len(Z) > 1:
        while len(Z) > 1:
            k = 0.5*max(X[l],Y[l])
            if np.max(np.delete(Z,l)) <  k and k < dtemp: 
                d = k
                break

            elif np.max(np.delete(Z,l)) >= k:
                if len(Z[Z>=k]) == len(np.where(np.where(Z >= k)[0] >= l)[0]): 
                    d = k
                    break
                else:
                    Z=Z[0:l]
                    X=X[0:l]
                    Y=Y[0:l]
                    l=np.argmax(Z)
                    dtemp = Z[l]             
                    if len(Z) == 1:
                        d = min(dtemp,0.5*max(X[l],Y[l]))
                        break
            else:
                d = dtemp
                break
    else:
        # If one of the persistence diagrams is a singleton 
        d = min(dtemp,0.5*max(X[0],Y[0]))

    return d

