from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import  UniformFloatHyperparameter
from autosklearn.pipeline.constants import (
    DENSE,
    UNSIGNED_DATA,
    SIGNED_DATA,
)
from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.constants import (
    DENSE,
    UNSIGNED_DATA,
    SIGNED_DATA
)


class CorrelationRemover( AutoSklearnPreprocessingAlgorithm):
    
    sf = None

    def __init__(self, alpha, **kwargs):
        self.alpha = float(alpha)
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def utils_fairlearn(cls, sf):
        sf =sf 

    def fit(self, X, y=None):
        from fairlearn.preprocessing import CorrelationRemover as FCR
        index = CorrelationRemover.sf
        if type(X) == tuple:
           X,y = X
        X = pd.DataFrame(X)
        self.alpha = float(self.alpha)
        self.preprocessor = FCR(
            sensitive_feature_ids=index, alpha=self.alpha
        )

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        
        if type(X) == tuple:
           X,y = X
        X = pd.DataFrame(X)
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "CorrelationRemover",
            "name": "CorrlelationRemover",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.0, upper=1, default_value=1
        )
        cs.add_hyperparameters([alpha])

        return cs
