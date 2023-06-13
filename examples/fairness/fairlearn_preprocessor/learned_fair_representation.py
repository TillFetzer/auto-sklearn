from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import  UniformFloatHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant   
)
from typing import Optional
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.constants import (
    DENSE,
    UNSIGNED_DATA,
    SIGNED_DATA
)
from examples.fairness.fairlearn_preprocessor.abstract_fair_preprocessor import  FairPreprocessor
from autosklearn.pipeline.components.base import AutoSklearnComponent
import numpy as np
import examples.fairness.only_label

class LFR(FairPreprocessor, AutoSklearnComponent):
    index_sf = None
    sf = None
    # n_prototypes, reconstruct_weight, target_weight, fairness_weight, tol, max_iter,
    def __init__(
        self,
        n_prototypes,
        reconstruct_weight,
        target_weight,
        fairness_weight,
        tol,
        max_iter,
        random_state=None,
        **kwargs,
    ):
        self.n_prototypes = self.n_prototypes
        self.reconstruct_weight = self.reconstruct_weight    
        self.target_weight = self.target_weight
        self.fairness_weight = self.fairness_weight
        self.tol = self.tol
        self.n_iter = self.n_iter
        self.random_state = random_state
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def utils_fairlearn(cls, index_sf):
        cls.index_sf = index_sf
        #cls.sf = sf

    def fit(self, X, Y=None):
        from aif360.sklearn.preprocessing import  LearnedFairRepresentations
        #X,Y = np.array(X), np.array(Y)
        # maybe type needs to transform
      
        self.preprocessor = LearnedFairRepresentations(
            prot_attr=LFR.index_sf,
            reconstruct_weight=self.reconstruct_weight,
            target_weight=self.target_weight,
            fairness_weight=self.fairness_weight,
            tol=self.tol,
            n_prototypes= self.n_prototypes,
            max_iter=self.n_iter,
            verbose=2,
            random_state=self.random_state
        )
        # patched something in aif360, not good
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X, y=None):
        if self.preprocessor is None:
            raise NotImplementedError()
        if y is None:
            return self.preprocessor.transform(X)
        y_pred = self.preprocessor.predict(X)
        examples.fairness.only_label.set_only_label(y_pred[0])
        return self.preprocessor.transform(X), y_pred
        #return np.array(self.preprocessor.transform(X)), np.array(self.preprocessor.predict(X)).astype("float")

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "LFR",
            "name": "CorrlelationRemover",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        #cahnge shortly the attribute to look on the higher picturec d
        n_prototypes = UniformIntegerHyperparameter("n_prototypes", 1, 200, default_value=50)
        reconstruct_weight = UniformFloatHyperparameter(
            "reconstruct_weight", 1e-4, 100, default_value=0.01, log=True
        )
        target_weight = UniformFloatHyperparameter(
            "target_weight", 1e-2, 100, default_value=0.5, log=True
        )
        fairness_weight = UniformFloatHyperparameter(
            "fairness_weight", 1e-10, 100, default_value=1e-5, log=True
        )
        tol = Constant("tol", 0)
        max_iter = UniformIntegerHyperparameter("max_iter",1000,15000, default_value=9000)
        cs.add_hyperparameters(
            [
                n_prototypes,
                reconstruct_weight,
                target_weight,
                tol,
                fairness_weight,
                max_iter
            ]
        )
        return cs
