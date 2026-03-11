import numpy as np




class OODDS():
    '''
    Out-of-distribution detection system (OODDS).

    References:
        - Testing for Outliers with Conformal p-values, Bates et al. (https://doi.org/10.48550/arXiv.2104.08279)

    This constructor initializes the class.
    All key attributes are initialized by default to be set during later stages of the process.

        
    Parameters
    ----------
    None

    
    Attributes
    ----------
    calOODscores : numpy.ndarray of shape (n_cal,)
        It saves the sorted vector in ascending order.
    '''
    def __init__(self):
        self.calOODscores = None


    def fit(self, calOODscores):
        '''
        Save OOD scores.
        

        Parameters
        ----------
        calOODscores : numpy.ndarray of shape (n_cal,)
        

        Returns
        -------
        None
            Modifies in place:
                - calOODscores : numpy.ndarray of shape (n_cal,)
                    It saves the sorted vector in ascending order.
        '''
        self.calOODscores = np.sort(calOODscores)


    def OODp(self, OODscore):
        '''
        Obtain a proxy for OOD probability for an instance.

        
        Parameters
        ----------
        OODscore : float

        
        Returns
        -------
        OODp : float
            OOD probability for the instance.
        '''
        idx = np.searchsorted(self.calOODscores, OODscore, side = 'right')
        return 100*idx / self.calOODscores.size
    

    def getOODps(self, OODscores):
        '''
        Obtain a proxy for OOD probabilities for instances.
    
        
        Parameters
        ----------
        OODscore : numpy.ndarray of shape (n,)

        
        Returns
        -------
        OODps : numpy.ndarray of shape (n,)
            OOD probabilities for the instances.
        '''
        idxs = np.searchsorted(self.calOODscores, OODscores, side = 'right')
        return 100*idxs / self.calOODscores.size