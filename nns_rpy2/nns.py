from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import numpy as np


class NNSRPY2Estimator(BaseEstimator):
    """An example of classifier"""

    def __init__(
            self,
            factor_2_dummy=True,
            order=None,
            stn=0.96,
            dim_red_method=None,
            tau=None,
            type=None,
            location="top",
            std_errors=False,
            confidence_interval=None,
            threshold=0,
            n_best=None,
            noise_reduction="mean",
            norm=None,
            dist="L2",
            ncores=None,
            multivariate_call=False
    ):
        """
        Called when initializing the classifier
        """
        # TODO: install R package automatically: https://rpy2.readthedocs.io/en/latest/introduction.html#getting-started

        self.factor_2_dummy = factor_2_dummy
        self.order = order
        self.stn = stn
        self.dim_red_method = dim_red_method
        self.tau = tau
        self.type = type
        self.location = location
        self.std_errors = std_errors
        self.confidence_interval = confidence_interval
        self.threshold = threshold
        self.n_best = n_best
        self.noise_reduction = noise_reduction
        self.norm = norm
        self.dist = dist
        self.ncores = ncores
        self.multivariate_call = multivariate_call

    def _retnull(self, value):
        if value is None:
            return robjects.NULL
        elif type(value) is bool:
            return robjects.vectors.BoolVector([False])
        return value

    def _nns_get_parameters(self, return_value=False):
        return {
            "factor_2_dummy": self._retnull(self.factor_2_dummy),
            "order": self._retnull(self.order),
            "stn": self._retnull(self.stn),
            "dim_red_method": self._retnull(self.dim_red_method),
            "tau": self._retnull(self.tau),
            "type": self._retnull(self.type),
            "location": self._retnull(self.location),
            "return_values": self._retnull(return_value),
            "plot": self._retnull(False),
            "plot_regions": self._retnull(False),
            "residual_plot": self._retnull(False),
            "std_errors": self._retnull(self.std_errors),
            "confidence_interval": self._retnull(self.confidence_interval),
            "threshold": self._retnull(self.threshold),
            "n_best": self._retnull(self.n_best),
            "noise_reduction": self._retnull(self.noise_reduction),
            "norm": self._retnull(self.norm),
            "dist": self._retnull(self.dist),
            "ncores": self._retnull(self.ncores),
            "multivariate_call": self._retnull(self.multivariate_call)
        }

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.r_NNS_ = importr('NNS')  # require(NNS)
        rpy2.robjects.numpy2ri.activate()
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, y=None):
        if not hasattr(self, "r_NNS_"):
            raise RuntimeError("You must train estimator before predicting data!")
        params = self._nns_get_parameters(True)
        params['point_est'] = X
        ret = self.r_NNS_.NNS_reg(self.X_, self.y_, **params)
        for i in range(len(ret.names)):
            if ret.names[i] == 'Point.est':
                return np.array(ret[i])
        raise RuntimeError("Can't find return Point.est value")


class NNSRPY2Classifier(NNSRPY2Estimator, ClassifierMixin):
    def __init__(self, **kwargs):
        kwargs['type'] = "CLASS"
        super().__init__(**kwargs)

    def predict_proba(self, X, y=None):
        # No not flor classification
        # Could run reg regression on it however
        # 0,1 classes like logistic regression works with type = NULL
        # Set type = NULL and noise.reduction = “mode”
        error, ret = None, None
        old_type, old_noise_reduction = self.type, self.noise_reduction
        try:
            self.type = None
            self.noise_reduction = "mode"
            ret = np.clip(self.predict(X, y), 0, 1)     # clip >=0, <=1
        except Exception as e:
            error = e
        # set back to original value
        self.type, self.noise_reduction = old_type, old_noise_reduction
        if error is not None:
            raise error
        return ret


class NNSRPY2Regressor(NNSRPY2Estimator, RegressorMixin):
    def __init__(self, **kwargs):
        kwargs['type'] = None
        super().__init__(**kwargs)


# TODO: check if we could export a cluster class (unsupervised algorithm)
class NNSRPY2Clusterer(NNSRPY2Estimator, ClusterMixin):
    def predict(self, X, y=None):
        if not hasattr(self, "r_NNS_"):
            raise RuntimeError("You must train estimator before predicting data!")
        params = self._nns_get_parameters(True)
        params['point_est'] = X
        ret = self.r_NNS_.NNS_reg(self.X_, self.y_, **params)
        for i in range(len(ret.names)):
            if ret.names[i] == 'Point.est.NNS.ID':     ## !! should return the NNS.ID of the Point.est value !!
                return np.array(ret[i])
        raise RuntimeError("Can't find return Point.est.NNS.ID value")


#if __name__ == '__main__':
#    model = NNSRPY2Clusterer()
#    x = np.array([1, 2, 3, 4])
#    x_new = x + 1
#    y = x ** 3
#    y_new = x_new + 1
#    model.fit(x, y)
#    print(model.predict(x, y))


