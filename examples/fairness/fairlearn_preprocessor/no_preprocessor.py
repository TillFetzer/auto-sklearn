from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.constants import (
    DENSE,
    UNSIGNED_DATA,
    SIGNED_DATA,
    SPARSE
)
from ConfigSpace.configuration_space import ConfigurationSpace


class  no_preprocessor(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors only remove the sensitive attributes"""
        for key, val in kwargs.items():
            setattr(self, key, val)
    

    def fit(self, X, Y=None):
        self.preprocessor = "no"
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        
        return X
    
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "no",
            "name": "no",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        return cs