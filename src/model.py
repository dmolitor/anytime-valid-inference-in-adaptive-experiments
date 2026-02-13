from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
from typing import Any

class Model(ABC):
    """
    A model class to support a variety of different models. This should accept
    an input matrix or dataframe `X` and response array `y`. This class must also
    implement the following classes:

    - fit(): This method should fit the model and return the fitted model
        object. The class should also record the fitted model under the
        self._fitted attribute. That way the user can access the fitted
        model directly or after the fact.
    - predict(X): This method should generate predicted values from the fitted
        model on a holdout set and return these predictions as a numpy array.
    """
    @abstractmethod
    def fit(self) -> Any:
        """
        Fit the underlying model and return the model object. This function
        should also store the model at the self._fitted attribute for later access.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame | npt.NDArray | Any) -> npt.NDArray[np.float64]:
        """
        Returns the predictions of the fitted model on a hold-out dataset. Returns
        predictions as a numpy array.
        """

class FastOLSModel(Model):
    def __init__(self, X: pd.DataFrame | npt.NDArray | Any, y: npt.NDArray):
        self._fitted = None
        self._X = X
        self._y = y

    def fit(self):
        self._fitted, _, _, _ = np.linalg.lstsq(self._X, self._y)
        return self

    def predict(self, X: pd.DataFrame | npt.NDArray):
        assert self._fitted is not None, "Attempting to predict before fitting a model"
        predictions = np.dot(X, self._fitted)
        return predictions

class LassoModel(Model):
    def __init__(self, X: pd.DataFrame | npt.NDArray | Any, y: npt.NDArray):
        self._fitted = None
        self._X = X
        self._y = y
    
    def fit(self, cv: int = 5, **kwargs):
        lasso = LassoCV(cv=cv, **kwargs)
        lasso.fit(self._X, self._y)
        self._fitted = lasso
        return lasso
    
    def predict(self, X):
        assert self._fitted is not None, "Attempting to predict before fitting a model"
        predictions = self._fitted.predict(X)
        return predictions

class LogitModel(Model):
    def __init__(self, X: pd.DataFrame | npt.NDArray | Any, y: npt.NDArray):
        self._fitted = None
        self._X = X
        self._y = y
    
    def fit(self, **kwargs):
        logit = sm.GLM(
            endog=self._y,
            exog=self._X,
            family=sm.families.Binomial(),
            **kwargs
        )
        logit = logit.fit()
        self._fitted = logit
        return logit
    
    def predict(self, X):
        assert self._fitted is not None, "Attempting to predict before fitting a model"
        predictions = self._fitted.predict(X)
        if isinstance(X, pd.DataFrame):
            predictions = predictions.to_numpy()
        return predictions

class OLSModel(Model):
    def __init__(self, X: pd.DataFrame | npt.NDArray | Any, y: npt.NDArray):
        self._fitted = None
        self._X = X
        self._y = y
    
    def fit(self, **kwargs):
        ols = sm.OLS(endog=self._y, exog=self._X, **kwargs)
        ols = ols.fit()
        self._fitted = ols
        return ols
    
    def predict(self, X):
        assert self._fitted is not None, "Attempting to predict before fitting a model"
        predictions = self._fitted.predict(X)
        if isinstance(X, pd.DataFrame):
            predictions = predictions.to_numpy()
        return predictions