"""xDAWN + Riemannian Geometry baseline classifier.

Classical BCI pipeline: xDAWN spatial filtering → covariance estimation →
Riemannian tangent space projection → logistic regression.

No GPU required. No hyperparameters to tune.
"""

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class XDawnRG:
    """xDAWN + Riemannian Geometry classifier.

    Input:  X (n_trials, n_channels, n_times), y (n_trials,)
    Output: probabilities (n_trials,) via predict_proba()
    """

    def __init__(self, nfilter=4):
        self.pipeline = Pipeline([
            ('xdawn_cov', XdawnCovariances(nfilter=nfilter, estimator='lwf')),
            ('tangent', TangentSpace(metric='riemann')),
            ('clf', LogisticRegression(max_iter=1000)),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
