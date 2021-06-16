import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from typing import Callable


class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    '''The class turns a function into sklearn pipeline compatible
    estimator.

    Example
    -------
    from sklearn import datasets
    from sklearn.pipeline import Pipeline

    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)


    @DataFrameFunctionTransformer
    def add_new_variable(input_df):
        return input_df.assign(new_variable=1)

    pipeline = Pipeline(steps=[('add_new_variable', add_new_variable)])
    pipeline.fit_transform(iris_df)

    '''
    def __init__(self, function: Callable):
        '''
        Parameters
        ----------
        function : callable
           The function to be wrapped by the wrapper.
        '''
        self.function = function

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        return self.function(X)


class DataFrameSKLearnTransformer(BaseEstimator, TransformerMixin):
    '''The class is a wrapper to SKLearn transformer which preserves the
    attributes of the input data frame.

    Example
    -------

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

    scaler = StandardScaler()
    data_scaler = DataFrameSKLearnTrnasformer(scaler)
    data_scaler.fit_transform(iris_df)
    '''

    def __init__(self, estimator: BaseEstimator):
        '''
        Parameters
        ----------
        estimator: Estimator
            A Sklearn estimator
        '''
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params) -> pd.DataFrame:
        index = X.index
        features = X.columns

        transformed_x = self.estimator.fit_transform(X)
        return pd.DataFrame(transformed_x, index=index, columns=features)
