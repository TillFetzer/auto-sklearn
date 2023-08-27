from examples.fairness.fairlearn_preprocessor import FairPreprocessor
from autosklearn.pipeline.components.base import AutoSklearnComponent
from ConfigSpace.configuration_space import ConfigurationSpace
from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
import pandas as pd
from autosklearn.pipeline.constants import (
    DENSE,
    UNSIGNED_DATA,
    SIGNED_DATA,
    SPARSE
)

class SensitiveAttributeRemover(FairPreprocessor, AutoSklearnComponent):
    index_sf = None

    def __init__(self, random_state):
        """This preprocessors only remove the sensitive attributes"""

    @classmethod
    def utils_fairlearn(cls, sf, index_sf):
        cls.sf = sf
        cls.index_sf = index_sf

    def fit(self, X, Y=None):
        self.preprocessor = "remover"
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        if self.preprocessor is None:
            raise NotImplementedError()
        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X)
            X.drop(columns=[SensitiveAttributeRemover.index_sf], inplace=True)
        else:
            X.drop(columns=[SensitiveAttributeRemover.sf], inplace=True)
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "SArm",
            "name": "SenstiveAtributeRemover",
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
