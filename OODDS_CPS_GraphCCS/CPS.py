import numpy as np
import pandas as pd
from math import lgamma
import warnings




class ConformalPredictiveSystem:
    """
    Univariate regression with an uncertainty wrapper using 
    Mondrian normalized conformal predictive systems (CPS).

    To assemble the CPS, apply the following methods in this order:
        1. Standard(): compute non-conformities and save inference model.
        2. Normalize(): in case a normalized CPS is sought.
        3. Mondrian(): turn the (normalized) standard CPS into Mondrian type,
        eventually with calibration-conditional validity.
        It must always be applied. In case there are actually no Mondrian taxonomies sought,
        pass one global category.

    The following additional methods are also available:
        - point_estimation()
        - scale(): obtain difficulties for instances.
        - categorize(): obtain Mondrian taxonomies for instances.
        - getCPDs(): obtain conformal predictive distributions.
        - getCRPSs() / CRPS(): calculate the continuous ranked probability score.
        - getPvalues() / Pvalue()
        - get CIs() / CI(): obtain confidence intervals.

    References:
        - Mondrian Conformal Predictive Distributions, Boström et al. (https://proceedings.mlr.press/v152/bostrom21a.html)
        - Accelerating difficulty estimation for conformal regression forests, Boström et al. (https://doi.org/10.1007/s10472-017-9539-9)
        - Evaluating different approaches to calibrating conformal predictive systems, Werner et al. (https://proceedings.mlr.press/v128/werner20a.html)
        - Testing for Outliers with Conformal p-values, Bates et al. (https://doi.org/10.48550/arXiv.2104.08279)

    This constructor sets up the basic configuration for the conformal regressor.
    All key attributes are initialized to default values (None or False) to be set during later stages
    of the process.

    
    Parameters
    ----------
    None 

    
    Attributes
    ----------
    inference_model : object
        The underlying regression model used for making predictions.
    nonconformities : dict
        Dictionary mapping each taxonomy label to its corresponding sorted-in-ascending-order (normalized) 
        non-conformity scores (these are the calibration residuals).
    oob : bool
        Whether calibration was performed using out-of-bag instances.
    OOBmask : numpy.ndarray array of shape (n_cal, ensemble_size)
        Matrix indicating whether a given instance was out-of-bag for a given ensemble estimator.
        Only relevant if oob = True.
    difficulty_model : object
        Object responsible for estimating instance difficulties.
    ccv_envelopes : dict
        Monte-Carlo envelope for each taxonomy to obtain calibration-conditional p-values.
        Only relevant when the CPS is conditioned on the calibrants.
    taxonomist : object
        Object responsible for categorizing instances into taxonomies for Mondrian conformal prediction.
    """
    def __init__(self):

        self.inference_model = None
        self.nonconformities = None
        self.oob = False
        self.OOBmask = None

        self.ccv_envelopes = None

        self.difficulty_model = None

        self.taxonomist = None


    def Standard(self, Xcal, y_cal, inference_model, oob = False):
        """
        Fit the standard conformal regressor and compute non-conformity scores.

        Uses the predictions from the inference_model on the provided calibration set (Xcal, y_cal) to compute
        non-conformity scores. Non-conformity scores are computed as residuals.

        
        Parameters
        ----------
        Xcal : numpy.ndarray of shape (n_cal_samples, n_features)
            Calibration data features.
        y_cal : numpy.ndarray of shape (n_cal_samples,)
            Calibration target values.
        inference_model : object
            A pre-fitted regression model used for making predictions with a method predict(X).
        oob : bool, default = False
            Whether to perform out-of-bag calibration. It allows to use as Xcal and y_cal the full training set.
            It requires that the inference model has attributes oob_prediction_ and
            estimators_samples_, as in, e.g., sklearn.ensemble.RandomForestRegressor.

            
        Returns
        -------
        None
            The method updates instance attributes in place:
                - inference_model : Passed pre-fitted regressor.
                - OOBmask : Matrix indicating whether a given instance was out-of-bag for a given ensemble estimator.
                - nonconformities : Residuals of the calibration instances as a np.ndarray of shape (n_cal,).
        """
        self.inference_model = inference_model

        if oob: # handling oob calibration
            self.oob = oob
            y_hat_cal = self.inference_model.oob_prediction_
            # create an boolean matrix indicating for each ensemble predictor which instances are out-of-bag
            self.OOBmask = np.ones((y_cal.size, len(self.inference_model.estimators_samples_)), dtype=bool)
            for estimator_idx, samples in enumerate(self.inference_model.estimators_samples_):
                self.OOBmask[samples, estimator_idx] = False  # in-bag
        
        else:
            y_hat_cal = self.inference_model.predict(Xcal)

        self.nonconformities = y_cal - y_hat_cal
    

    def Normalize(self, Xcal, difficulty_model):
        """
        Normalize the standard conformal regressor.
        Uses the difficuly_model's predictions on the provided calibration set (Xcal)
        to compute difficulties for each calibration instance. For normalization,
        the calibration residuals are divided by their corresponding estimated difficulty.

        
        Parameters
        ----------
        Xcal : numpy.ndarray of shape (n_cal_samples, n_features)
            Calibration data features.
        difficulty_model : object
            A pre-fitted difficulty model used for making difficulty estimations with a method complicate(X).
        
            
        Returns
        -------
        None
            The method updates instance attributes in place:
                - difficulty_model : Passed pre-fitted difficulty model.
                - nonconformities : Non-conformity scores are divided by their corresponding difficulty.
        """
        self.difficulty_model = difficulty_model
        cal_difficulties = self.difficulty_model.complicate(Xcal)
        
        self.nonconformities /= cal_difficulties
    

    def simes_envelope(self, n_cal, delta = 0.05):
        '''
        Helper function to compute the Simes envelope.
        '''
        k = n_cal // 2
        bs = np.full(n_cal, np.nan)
        # use factorial in log-space
        denominator = lgamma(n_cal+1) - lgamma(n_cal - k + 1) # +1 because gamma function of n = (n-1)! !!!
        for i in range(1, n_cal + 1):
            if i >= k:
                numerator = lgamma(i + 1) - lgamma(i - k + 1)
                quotient = np.exp((numerator - denominator) / k) # go back from log-space to original scale
            else: # factorial of negative values
                quotient = 0
            bs[n_cal - i] = 1.0 - (delta ** (1.0 / k)) * quotient
        return bs
    

    def asymptotic_envelope(self, n_cal, delta = 0.05):
        '''
        Helper function to compute the asymptotic envelope.
        '''
        as_fun = -np.log(-np.log(1-delta)) + 2*np.log(np.log(n_cal)) + 0.5 * np.log(np.log(np.log(n_cal))) - 0.5 * np.log(np.pi)
        as_fun /= np.sqrt(2*np.log(np.log(n_cal)))
        i = np.arange(1,n_cal+1)
        return np.minimum(i/n_cal + as_fun * np.sqrt(i*(n_cal-i))/(n_cal*np.sqrt(n_cal)), 1)
    

    def hybrid_envelope(self, n_cal, delta_hat, delta = 0.05):
        """
        Pointwise minimum of Simes and asymptotic envelopes.

        Code based on:
            - https://github.com/msesia/conditional-conformal-pvalues
                - Linear extension based on the tangent at n // 2
        """
        ### simes
        simes_bs = self.simes_envelope(n_cal, delta)
        ### asymptotic bound
        asymptotic_bs = self.asymptotic_envelope(n_cal, delta_hat)
        # At large p-values (i >= n /2) the hybrid envelope is typically governed by the asymptotic envelope.
        # Those p-values are not relevant for rejections of instances, 
        # but the crossing check still spends error budget δ there, including i > n/2.
        # To avoid wasting δ on the upper half (and thus having to inflate the bound where it matters),
        # we loosen the envelope for i >= n/2 by continuing it with the tangent line at i = n/2.
        # This linear extension lies above the original asymptotic envelope, reducing crossings in the upper half.
        i_linear = n_cal // 2
        slope = (asymptotic_bs[i_linear-1]-asymptotic_bs[i_linear-2]) # / 1
        i = np.arange(1,n_cal+1)
        asymptotic_bs[i_linear:] = asymptotic_bs[i_linear-1] + slope * (i[i_linear:]-i_linear)
        return np.minimum(simes_bs, asymptotic_bs)


    def estimate_delta_hat(self, n_cal, delta = 0.05, n_mc = 10000, tol = 1e-6, seed = None):
        '''
        Estimate the finite-sample correction for the asymptotic envelope.

        Code based on:
            - https://github.com/msesia/conditional-conformal-pvalues
                - General structure of the function
                - Helper function to calculate P[u(1) <= b1, ..., u(n) <= b(n)]


        Parameters
        ----------
        n_cal : int
            Number of calibrant instances.
        delta : float,  default = 0.05
            With a probability of at least 1 − δ, the p-values will be conservative.
        n_mc : int, default = 10000
            Number of independent random draws from the uniform distribution for Monte-Carlo.
        tol : float, default = 1e-6
            Bisection stopping width.
        seed : int, default = None
            Random state seed.

            
        Returns
        -------
        delta_hat : float
            Estimate for the finite-sample correction of delta for the asymptotic envelope.
        '''
        # P[u(1) <= b1, ..., u(n) <= b(n)] >= 1 - δ (1)
        # draw the us
        np.random.seed(seed)
        U = np.random.uniform(size=(n_mc,n_cal))
        U = np.sort(U,axis=1)

        # The asymptotic envelope is valid when n_cal --> ∞.
        # Our goal is to find a δhat for the asymptotic envelope due to the finite sample (calibration).
        # Since as_fun is a decreasing function with δ (tighter envelope),
        # we would like to find the largest possible estimated δhat 
        # that guarantees the calibration-conditional validity (1).
        # We search δhat using a bisection algorithm in the range [1e-6, 1 - 1e-6]
        # (1e-6 to guard against the extremes where as_fun is not well defined due to the ln).
        # This means we evaluate if the (1) holds for both extremes; if that is the case,
        # we can increase δhat by halving the range.
        # We do this until (1) doesn't hold anymore.
        def calculateP1(delta_hat):
            '''
            Helper function to calculate if (1) still holds.
            '''
            h_envelope = self.hybrid_envelope(n_cal, delta_hat=delta_hat, delta = delta)
            crossings = np.sum(U>h_envelope,1)
            P1 = np.mean(crossings>0)
            return P1
        delta_hat0 = 1e-6 # lower extreme (broader envelope)
        delta_hat1 = 1-1e-6 # higher extreme (tighter envelope)
        while delta_hat1-delta_hat0 > tol: # bisection stopping width
            _lambda = (delta_hat1 - delta_hat0)/2 # bisection
            # now we need to decide to which end we give the _lambda
            P1 = calculateP1(delta_hat0+_lambda)
            if P1>delta:
                # if P1 > δ, that means we need to broaden the envelope
                # broaden = lower δhat
                delta_hat1 = delta_hat1-_lambda
            else:
                # if P1 <= δ, that means we can tighten the envelope
                # tighten = higher δhat
                delta_hat0 = delta_hat0+_lambda
        return delta_hat0


    def Mondrian(self, cal_taxonomies=None, X=None, taxonomist=None, ccv = False, **MCkwargs):
        """
        Fit the Mondrian categorizer and compute taxonomy-specific normalized non-conformity scores.

        This method assigns each calibration instance to a taxonomy label using the provided taxonomist,
        and then stores the normalized non-conformity scores for each taxonomy in the nonconformity dictionary,
        with taxonomy labels as keys. The normalized non-conformity scores for each taxonomy are sorted 
        in ascending order after prepending -∞ and appending ∞.

        Note that this method has to be used even if a standard CPS is sought. In that case, a global single taxonomy
        can be passed to cal_taxonomies.

        
        Parameters
        ----------
        cal_taxonomies : numpy.ndarray of shape (n_cal,), default = None
            Taxonomy labels for each calibrant. Only needed if X is None.
        X : numpy.ndarray of shape (n_cal, n_features), default = None
            Calibration data features for taxonomist. Only needed if cal_taxonomies is None.
        taxonomist : object
            A fitted taxonomist with a method taxidermize(X) that assigns taxonomy labels to calibration instances in X.
            Only needed if cal_taxonomies is None.
        ccv : bool, default = False
            Whether to pre-compute correction to obtain calibration-conditional validity.
        MCkwargs : -
            Any parameters passed to the Monte-Carlo simulation function.

            
        Returns
        -------
        None
            Updates the following instance attributes in place:
                - taxonomist : the provided taxonomist.
                - nonconformities : dictionary mapping taxonomy labels to their corresponding scores.
                - ccv_envelopes : Monte-Carlo envelopes for each taxonomy. Only if ccv = True.
        """
        if cal_taxonomies is not None:
            taxonomies = cal_taxonomies
        else:
            self.taxonomist = taxonomist # fitted taxonomist to produce Mondrian classes
            taxonomies = self.taxonomist.taxidermize(X) # taxonomy label for each calibration instance

        partitioned_nonconformities = {}
        unique_taxonomies = np.unique(taxonomies)
        ccv_envelopes = {} if ccv else None # calibration-conditional validity correction
        # create a dictionary with an entry per taxonomy containing the calibration scores
        for taxonomy in unique_taxonomies:
            cal_taxonomy_partners = taxonomies == taxonomy
            nonconformities_sorted = np.concatenate(
                (
                    [-np.inf], 
                    np.sort(self.nonconformities[cal_taxonomy_partners]),
                    [np.inf]
                )
            )
            partitioned_nonconformities[taxonomy] = nonconformities_sorted

            if ccv: 
                n_cal = sum(cal_taxonomy_partners)
                delta_hat = self.estimate_delta_hat(n_cal, **MCkwargs)
                delta = 0.05 if MCkwargs.get('delta') is None else MCkwargs.get('delta')
                ccv_envelopes[taxonomy] = np.concatenate(
                    ([0], self.hybrid_envelope(n_cal, delta = delta, delta_hat=delta_hat), [1])
                )

        self.nonconformities = partitioned_nonconformities
        self.ccv_envelopes = ccv_envelopes if ccv else None
        
  
    def point_estimation(self, X, aggregation_function = np.mean, seed = None):
        """
        Perform point estimation for new instances.

        
        Parameters
        ----------
        X : numpy.ndarray of shape (n, n_features)
            Data features.
        aggregation_function : function, default = numpy.mean
            Function to aggregate predictions from each ensemble estimator
        seed : int, default = None
            Random state for selecting a subset of ensemble estimators.

            
        Returns
        -------
        predictions : numpy.ndarray of shape (n,)
            1D NumPy array containing the point prediction for each instance.
        """
        if self.oob:
            # for each index select a calibration instance
            # use the predictors that had that instance out-of-bag to do estimation
            np.random.seed(seed)
            subsets = np.random.choice(np.arange(self.OOBmask.shape[0]), size = X.shape[0], replace = True)
            predictions = np.full(X.shape[0], fill_value=np.nan)
            for query_idx, subset in enumerate(subsets):
                individual_predictions = [
                    self.inference_model.estimators_[estimator_idx].predict(X[query_idx].reshape(1, -1))[0] 
                    for estimator_idx in np.where(self.OOBmask[subset])[0]
                ]
                predictions[query_idx] = aggregation_function(individual_predictions)
               
        else:
            predictions =  self.inference_model.predict(X)            

        return predictions


    def scale(self, X):
        """
        Obtain difficulties for instances using the fitted difficulty model
        by applying it to X.
        

        Parameters
        ----------
        X : numpy.ndarray of shape (n, n_features)
            Array of features for the instances to be complicated.
        
            
        Returns
        -------
        numpy.ndarray of shape (n,)
            Array of difficulties for each instance in X.
        """
        return self.difficulty_model.complicate(X)


    def categorize(self, X):
        """
        Categorize instances using the fitted taxonomist.

        This method assigns taxonomy labels to the provided instances X by applying the
        taxidermize method of the fitted taxonomist.

        
        Parameters
        ----------
        X : numpy.ndarray of shape (n, n_features)
            Array of features for the instances to be categorized.

            
        Returns
        -------
        numpy.ndarray of shape (n,)
            Array of taxonomy labels for each instance in X.
        """
        return self.taxonomist.taxidermize(X)


    def getCPDs(self, y_hat, difficulties, taxonomies):
        """
        Return onformal predictive distributions (CPDs) for instances.

        
        Parameters
        ----------
        y_hat : numpy.ndarray of shape (n,)
            Point predictions for the instances.
        difficulties : numpy.ndarray of shape (n,)
            1D NumPy array of difficulties for the instances.
            In case the CPS is not normalized, pass 1s.
        taxonomies : numpy.ndarray of shape (n,)
            1D NumPy array of taxonomy labels for the instances.

            
        Returns
        -------
        A list (length n), where each element is a 1D NumPy array of shape (taxonomy_size + 2, ) 
        representing the CPD for that instance.
        """
        conformal = [] # to store the CPD for each instance  
        for query_idx in range(y_hat.size):
            taxonomy = taxonomies[query_idx]
            if taxonomy in self.nonconformities:
                conformal.append(self.nonconformities.get(taxonomy)*difficulties[query_idx] + y_hat[query_idx])
            else:
                conformal.append(None)
        return conformal


    def getCRPSs(self, y, y_hat, difficulties, taxonomies, verbose = True):
        """
        Obtain the Continuous Ranked Probability Score (CRPS) for each instance in y.

        
        Parameters
        ----------
        y : numpy.ndarray of shape (n,)
            Ground truth target values.
        y_hat : int or float
            Point predictions for the instances.
        difficulties : numpy.ndarray of shape (n,)
            1D NumPy array of difficulties for the instances.
            In case the CPS is not normalized, pass 1s.
        taxonomies : numpy.ndarray of shape (n,)
            1D NumPy array of taxonomy labels for the instances.
        verbose : bool, default = True
            Whether to output integration warning print-outs.

            
        Returns
        -------
        crps_values : numpy.ndarray of shape (n,)
            1D NumPy array containing the CRPS for each instance.
        """
        crps_values = []
        # loop over each instance to compute CRPS
        for query_idx in range(y.size):
            crps_values.append(self.CRPS(y[query_idx], y_hat[query_idx], difficulties[query_idx], taxonomies[query_idx], verbose = verbose))
        return np.asarray(crps_values)
    

    def CRPS(self, y_i, y_hat_i, difficulty, taxonomy, verbose = True):
        """
        Compute the Continuous Ranked Probability Score (CRPS) for a single instance.

        This method constructs a crisp cumulative distribution function (CDF) and then 
        approximates the CRPS by numerically integrating the squared error between the
        crisp CDF and the ideal step function at the observed value y_i.

        Code based on:
            - Conformal Prediction in Python with crepes, Boström et al. (https://proceedings.mlr.press/v230/bostrom24a.html)
            (https://github.com/henrikbostrom/crepes)
                - Vectorization of the integral calculation

        
        Parameters
        ----------
        y_i : int or float
            Ground truth target value.
        y_hat_i : int or float
            Point prediction for the instance.
        difficulty : int or float
            Difficulty for the instance.
            In case the CPS is not normalized, pass 1.
        taxonomy : int
            Taxonomy label for the instance.
        verbose : bool, default = True
            Whether to output integration warnings.

            
        Returns
        -------
        crps_value : float
            The computed CRPS value for the instance.
        """
        if taxonomy in self.nonconformities:

            ### construct a crisp CDF
            taxonomy_scores = self.nonconformities.get(taxonomy)[1:-1]
            taxonomy_size = taxonomy_scores.size
            Fs = np.asarray([i+1 for i in range(taxonomy_size)]) / (taxonomy_size)
            # select only last index of each tie block
            counts = np.unique(taxonomy_scores, return_counts=True)[1]
            last_idxs = np.cumsum(counts) - 1
            shifted_scores = taxonomy_scores[last_idxs]*difficulty + y_hat_i
            Fs = Fs[last_idxs][:-1]
            
            ### find index k such that cal_scores[k] < y_i <= cal_scores[k+1]
            k = np.searchsorted(shifted_scores, y_i, 'left') - 1

            ### vectorized integration
            dx = np.diff(shifted_scores)
            wannabe0 = Fs ** 2 # p-values of the dxs that end up below y_i that should be 0 for a perfect prediction
            wannabe1 = (Fs - 1) ** 2 # # p-values of the dxs that end up above y_i that should be 1 for a perfect prediction
            if k == -1: # y_i sits between -∞ and the first finite calibrant or at the first finite calibrant
                if verbose:
                    warnings.warn(f'CRPS was calculated for instance with value {y_i} between -∞ and {shifted_scores[0]}.')
                return np.dot(wannabe1, dx) + shifted_scores[0] - y_i # Fs = 0 --> (0 -1) ** 2 = 1
            elif k == shifted_scores.size - 1: # y_i sits between the last finite calibrant and ∞
                if verbose:
                    warnings.warn(f'CRPS was calculated for instance with value {y_i} between {shifted_scores[-1]} and ∞.')
                return np.dot(wannabe0, dx) + y_i - shifted_scores[-1] # Fs = 1 --> (1) ** 2 = 1
            else:
                return np.dot(wannabe0[:(k+1)], dx[:(k+1)]) + \
                    Fs[k] ** 2 * (y_i - shifted_scores[k]) + \
                    (Fs[k] - 1) ** 2 * (shifted_scores[k+1] - y_i) + \
                    np.dot(wannabe1[(k+1):], dx[(k+1):])
            
        else:
            return np.nan


    def getPvalues(self, y, y_hat, difficulties, taxonomies, seed = None):
        """
        Obtain smoothed central p-values for each instance in y,
        for the H0 that the ys are drawn from the respective CPDs.
        

        Parameters
        ----------
        y : numpy.ndarray of shape (n,)
            Ground truth target values.
        y_hat : int or float
            Point predictions for the instances.
        difficulties : numpy.ndarray of shape (n,)
            1D NumPy array of difficulties for the instances.
            In case the CPS is not normalized, pass 1s.
        taxonomies : numpy.ndarray of shape (n,)
            1D NumPy array of taxonomy labels for the instances.
        seed : int
            Random state for fuzziness.

            
        Returns
        -------
        p_values : numpy.ndarray of shape (n,)
            1D NumPy array containing the p_values for each instance.
        """
        np.random.seed(seed)
        taus = np.random.rand(y.size) # fuzziness sampling from U(0, 1)
        p_values = []
        # loop over each instance to compute smoothed p-values
        for query_idx in range(y.size):
            p_values.append(self.Pvalue(y[query_idx], y_hat[query_idx], difficulties[query_idx], taxonomies[query_idx], taus[query_idx]))
        
        return np.asarray(p_values)


    def Pvalue(self, y_i, y_hat_i, difficulty, taxonomy, tau):
        """
        Obtain the smoothed central p-value for instance y_i,
        for the H0 that y_i is drawn from the respective CPD.
        

        Parameters
        ----------
        y_i : int or float
            Ground truth target value.
        y_hat_i : int or float
            Point prediction for the instance.
        difficulty : int or float
            Difficulty for the instance.
            In case the CPS is not normalized, pass 1.
        taxonomy : int
            Taxonomy label for the instance.
        tau : float
            Fuzziness value drawn from U(0,1).

            
        Returns
        -------
        p_value : float
            p-value
        """
        if taxonomy in self.nonconformities:

            shifted_scores = self.nonconformities.get(taxonomy)*difficulty + y_hat_i
            i = np.searchsorted(shifted_scores, y_i, side = 'right') - 1
            if np.isclose(shifted_scores[i], y_i): # existing ties
                tie_idxs = np.where(np.isclose(shifted_scores, shifted_scores[i]))[0]
                p_value = (tie_idxs[0] - 1 + (tie_idxs[-1]- tie_idxs[0] + 2) * tau) / (shifted_scores.size - 1)
            else: # no ties
                p_value = (i + tau) / (shifted_scores.size - 1)
            p_value = 2 * min(p_value, 1 - p_value)

            if self.ccv_envelopes is not None: # calibration-conditional validity
                p_value = self.ccv_envelopes.get(taxonomy)[np.ceil((shifted_scores.size - 1) * p_value).astype(int)]

            return 100 * p_value
        
        else:
            return np.nan
        

    def getCIs(self, y_hat, difficulties, taxonomies, confidence):
        """
        Obtain equal-tailed confidence intervals for instances at a given confidence level.

        For each instance, this method computes the lower and upper bounds of the 
        equal-tailed confidence interval.

        
        Parameters
        ----------
        y_hat : numpy.ndarray of shape (n,)
            Point predictions for the instances.
        difficulties : numpy.ndarray of shape (n,)
            1D NumPy array of difficulties for the instances.
            In case the CPS is not normalized, pass 1s.
        taxonomies : numpy.ndarray of shape (n,)
            1D NumPy array of taxonomy labels for the instances.
        confidence : float
            The desired confidence level (e.g., 0.95 for 95% confidence).

            
        Returns
        -------
        numpy.ndarray of shape (n, 2)
            A 2D array where each row contains [lower bound, upper bound] for each instance's 
            equal-tailed confidence interval.
        """
        n_query = y_hat.size

        # initialize arrays to store the lower and upper bounds
        lowers = np.full(n_query, np.nan)
        uppers = np.full(n_query, np.nan)

        for query_idx in range(n_query):
            lowers[query_idx], uppers[query_idx] = self.CI(y_hat[query_idx], difficulties[query_idx], taxonomies[query_idx], confidence)

        return np.column_stack((lowers, uppers))
    

    def CI(self, y_hat_i, difficulty, taxonomy, confidence):
        """
        Compute the equal-tailed confidence interval for a single instance.
        This method computes the lower and upper bounds of the equal-tailed confidence interval.

        
        Parameters
        ----------
        y_hat_i : int or float
            Point prediction for the instance.
        difficulty : int or float
            Difficulty for the instance.
            In case the CPS is not normalized, pass 1.
        taxonomy : int
            Taxonomy label for the instance.
        confidence : float
            The desired confidence level (e.g., 0.95 for 95% confidence).

            
        Returns
        -------
        lower bound: float
            The lower bound of the equal-tail confidence interval.
        upper bound: float
            The upper bound of the equal-tail confidence interval.
        """

        if taxonomy in self.nonconformities:

            scores = self.nonconformities.get(taxonomy)

            alphaL = (1 - confidence) / 2 # lower tail; 1 - alphaL upper tail
                    
            if self.ccv_envelopes is not None: # use ccv envelope
                envelope = self.ccv_envelopes.get(taxonomy)
                alphaL = (np.searchsorted(envelope, alphaL, side = 'right') - 1) / (scores.size - 1)
            
            # select the sorted-in-ascending-order calibration score corresponding to the alpha quantile (lower bound)
            lower_idx = int(alphaL * (scores.size - 1))
            # select the sorted-in-ascending-order calibration score corresponding to the 1 - alpha quantile (upper bound)
            upper_idx = int(np.ceil((1 - alphaL) * (scores.size - 1)))

            if (upper_idx == scores.size - 1 or lower_idx == 0) and confidence != 1:   
                warnings.warn(f'The number of calibrants was too small for taxonomy {taxonomy} to \
                    obtain a finite sample-adjusted quantile at the given confidence.')
                
            lower = scores[lower_idx]*difficulty + y_hat_i
            upper = scores[upper_idx]*difficulty + y_hat_i

            return lower, upper
        

        else:
            return np.nan, np.nan



class taxonomist():
    """
    Trivial taxonomist for building Mondrian classes for ConformalPredictiveSystem.Mondrian().
    It discretizes a continuous variable.

    This constructor initializes the taxonomist class.
    All key attributes are initialized by default to None or False to be set during later stages of the process.

    
    Parameters
    ----------
    None 

    
    Attributes
    ----------
    edges : numpy.ndarray of shape (partition_size+1,)
        Bin edges.
    """
    def __init__(self):
        self.edges = None
        
        
    def fit(self, x_train, quantile_discretization = True, minimum_n_taxonomy_calibrants = 1000, partition_size = None):
        """
        Fit discretization.
        

        Parameters
        ----------
        x_train : numpy.ndarray of shape (n_train,)
            Target values on the training set.
        quantile_discretization : bool, default = True
            Wheter to form the taxonomies using quantile binning. This results in taxonomies of approximately
            the same size (number of calibrants). Othewise, equal-width taxonomies will be fit with a minimum
            number of calibrants.
        minimum_n_taxonomy_calibrants : int, default = 1000
            Minimum number of calibrants per taxonomy required.
        partition_size : int, default = None
            Size of the partition to be found. If None, a partition will be searched that fits the requirement
            of minimum_n_taxonomy_calibrants.
        
            
        Returns
        -------
        None
            Updates the following attributes:
                - edges
        """
        if quantile_discretization: # quantile binning

            if partition_size is None:
                partition_size = int(np.floor(x_train.size / minimum_n_taxonomy_calibrants))

            self.edges = pd.qcut(x_train, q=partition_size, labels=False, retbins=True, duplicates='drop')[1]
            self.edges[0], self.edges[-1] = -np.inf, np.inf


        else: # equal-width

            if partition_size is None:
                # search for the maximum number of equal-width taxonomies so that all contain at least minimum_n_taxonomy_calibrants calibrants
                init_partition_size = 2
                taxonomies = pd.cut(x_train, bins = init_partition_size, labels=False, retbins = False, duplicates='drop')
                while sum(np.unique(taxonomies, return_counts=True)[1] < minimum_n_taxonomy_calibrants) == 0:
                    init_partition_size += 1
                    taxonomies = pd.cut(x_train, bins = init_partition_size, labels=False, retbins = False, duplicates='drop')
                # step back to the last valid partition size
                partition_size = init_partition_size - 1

            self.edges = pd.cut(x_train, bins = partition_size, labels=False, retbins = True, duplicates='drop')[1]
            self.edges[0], self.edges[-1] = -np.inf, np.inf


    def taxidermize(self, x):
        '''
        Obtain taxonomy labels.
        

        Parameters
        ----------
        x : numpy.ndarray of shape (n,)
            Target target values on the set.
        
            
        Returns
        -------
        numpy.ndarray of shape(n,)
            Taxomy labels for each instance.
        '''
        return pd.cut(x, self.edges, labels=False, duplicates='raise')
    


class DifficultyModel():
    """
    Trivial difficulty model for generating difficulties.

    This constructor initializes the class.
    All key attributes are initialized by default to None or False to be set during later stages of the process.

    
    Parameters
    ----------
    None 

    
    Attributes
    ----------
    sigma : Python function
        Function to compute difficulties.
        It must return a vector of difficulties.
    """
    def fit(self, sigma):
        """
        Fit difficulty model by passing a function to compute them.
        

        Parameters
        ----------
        sigma : function
            Function to compute difficulties.
            It must return a vector of difficulties.

            
        Returns
        -------
        None
            Updates the following attributes:
                - sigma
        """
        self.sigma = sigma


    def complicate(self, x):
        """
        Calculate difficulties.
        

        Parameters
        ----------
        x : -
            Input for function sigma.
        
            
        Returns
        -------
        difficulties : numpy.ndarray of shape (n,)
            Difficulties for each instance.
        """
        return self.sigma(x)