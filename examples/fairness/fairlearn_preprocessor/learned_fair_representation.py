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


class LFR(AutoSklearnPreprocessingAlgorithm):
    index_sf = None
    # n_prototypes, reconstruct_weight, target_weight, fairness_weight, tol, max_iter,
    def __init__(
        self,
        n_prototypes,
        reconstruct_weight,
        target_weight,
        fairness_weight,
        tol,
        max_iter, 
        predict_y,
        **kwargs,
    ):
        self.n_prototypes = n_prototypes
        self.reconstruct_weight = reconstruct_weight
        self.target_weight = target_weight
        self.fairness_weight = fairness_weight
        self.tol = tol
        self.n_iter = max_iter
        self.predict_y = predict_y
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def utils_fairlearn(cls, index_sf):
        cls.index_sf = index_sf

    def fit(self, X, y=None):
        from aif360.sklearn.preprocessing import  LearnedFairRepresentations

        # maybe type needs to transform
        self.preprocessor = LearnedFairRepresentations(
            prot_attr=LFR.index_sf,
            reconstruct_weight=self.reconstruct_weight,
            target_weight=self.target_weight,
            fairness_weight=self.fairness_weight,
            tol=self.tol,
            n_prototypes= self.n_prototypes,
            max_iter=self.n_iter,
        )
        # patched something in aif360, not good
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        if self.predict_y:
            return self.preprocessor.transform(X), self.preprocessor.predict(X)
        else:
            return self.preprocessor.transform(X)

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
        n_prototypes = UniformIntegerHyperparameter("n_prototypes", 1, 100, default_value=50)
        reconstruct_weight = UniformFloatHyperparameter(
            "reconstruct_weight", 0.0001, 1, default_value=0.01, log=True
        )
        target_weight = UniformFloatHyperparameter(
            "target_weight", 0.01, 100, default_value=30, log=True
        )
        fairness_weight = UniformFloatHyperparameter(
            "fairness_weight", 0.5, 500, default_value=50, log=True
        )
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1, default_value=1e-4)
        predict_y = CategoricalHyperparameter("predict_y", [True])
        max_iter = Constant("max_iter",6000)
        cs.add_hyperparameters(
            [
                n_prototypes,
                reconstruct_weight,
                target_weight,
                tol,
                fairness_weight,
                max_iter,
                predict_y
            ]
        )
        return cs
