import numpy as np
from autosklearn.pipeline.components.base import AutoSklearnComponent
from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.constants import (
    DENSE,
    UNSIGNED_DATA,
    SIGNED_DATA,
    SPARSE
)
from examples.fairness.fairlearn_preprocessor import FairPreprocessor
from ConfigSpace.configuration_space import ConfigurationSpace


class  NoFairPreprocessor(FairPreprocessor, AutoSklearnComponent):
    index_sf = []
    def __init__(self, **kwargs):
        """This preprocessors samples that the data is fair"""
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    @classmethod
    def utils_fairlearn(cls, index_sf):
        cls.index_sf.append(index_sf)

    def fit(self, X, Y=None):
        self.preprocessor = "no"
        self.fitted_ = True
        return self
    def transform(self, X, y=None):
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