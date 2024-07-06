import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern

from bayes_opt.util import acq_max, ensure_rng

RANDOM_STATE = 10154545

class SimpleBayesOptimizer(object):

    def __init__(self, acq, kernel):
        # Internal GP regressor

        self._random_state =  np.random.RandomState(RANDOM_STATE)

        if not kernel:
            kernel = Matern(nu=2.5)

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=500,
            random_state=self._random_state,
        )

        self._acq = acq

        self.constraint = None
    
    def fit(self, X, y):
        self._gp.fit(X, y)

    def suggest(self, maxSoFar, maxSoFarParams, bounds):
        # Finding argmax of the acquisition function.
        suggestion = acq_max(ac=self._acq.utility,
                             gp=self._gp,
                             y_max=maxSoFar,
                             y_max_params=maxSoFarParams,
                             bounds=bounds,
                             constraint=self.constraint,
                             random_state=self._random_state,
                             )
        return suggestion

    