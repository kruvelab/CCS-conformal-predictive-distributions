import numpy as np
import cvxpy as cp




class PSVR:
    """
    Potential Support Vector Machine for regression (P-SVR).
    From "Support Vector Machines for Dyadic Data, Hochreiter et al. (https://doi.org/10.1162/neco.2006.18.6.1472)".


    Parameters
    ----------
    C : float, default = 1.0
        C-regularization parameter.
    epsilon : float, defalt = 0.1
        L1-regularization parameter.
    stdev_threshold : float or int, default = 1
        Columns with standard deviation <= stdev_threshold will be dropped.

        
    Attributes
    ----------
    C : float
        C-regularization.
    epsilon : float
        L1-regularization.
    stdev_threshold : float or int
        Columns with standard deviation <= stdev_threshold will be dropped.
    alpha : np.ndarray of shape (n_background,)
    b : float
        Offset.
    col_means : np.ndarray of shape (n_background,)
        Training column means.
    col_stds : np.ndarray of shape (n_background,)
        Training column standard deviations.
    non_constants : np.ndarray of shape (n_background,)
        Columns in the training set whose standard deviation > stdev_threshold.
    """
    def __init__(self, C=1.0, epsilon=0.1, stdev_threshold = 1):
        self.C = C
        self.epsilon = epsilon
        self.stdev_threshold = stdev_threshold
        self.alpha = None
        self.b = None
        self.col_means = None
        self.col_stds = None
        self.non_constants = None


    def fit(self, Xtrain, y_train):
        '''
        Fit the P-SVM.

        Parameters
        ----------
        Xtrain : np.ndarray of shape (n_train, n_background)
        y_train : np.ndarray of shape (n_train,)
        '''
        K = Xtrain.copy()

        n, _ = K.shape

        # mean 0 and variance 1/n
        col_means = K.mean(axis=0)
        col_stds = K.std(axis=0, ddof=1)
        non_constants = col_stds > self.stdev_threshold # drop constant columns
        col_stds = col_stds * np.sqrt(n)
        K = (K[:, non_constants] - col_means[non_constants]) / col_stds[non_constants]

        Q = K.T @ K
        alpha = cp.Variable(sum(non_constants)) # cvxpy variable alpha (p,)
        linear_vec = - y_train @ K
        f2minimize = ( # function to minimize
            0.5 * cp.quad_form(alpha, Q)
            + linear_vec @ alpha
            + self.epsilon * cp.norm1(alpha)
        )
        constraints = [ # contraints for minimization
            alpha <= self.C,
            alpha >= -self.C,
        ]

        prob = cp.Problem(cp.Minimize(f2minimize), constraints)
        prob.solve()
        # By default CVXPY calls the solver most specialized to the problem type.
        # If the problem is a QP, CVXPY will use OSQP.
        # (https://www.cvxpy.org/tutorial/solvers/index.html)

        if alpha.value is None:
            raise Exception("P-SVR optimization failed or did not converge.")

        self.alpha = alpha.value
        self.b = y_train.mean()
        self.col_means = col_means
        self.col_stds = col_stds
        self.non_constants = non_constants


    def predict(self, X):
        # apply the same column standardization
        Knew = X.copy()
        Knew = (Knew[:, self.non_constants] - self.col_means[self.non_constants]) / self.col_stds[self.non_constants]

        return Knew @ self.alpha + self.b