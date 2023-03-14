# this really should be implemented in other files, but right now, is here for testing
# and quick access

import autosklearn
import autosklearn.pipeline.components.classification as classification
import autosklearn.pipeline.components.data_preprocessing as pre_processing
from typing import Optional
import numpy as np
import fairlearn
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import (
    DENSE,
    PREDICTIONS,
    SPARSE,
    UNSIGNED_DATA,
    SIGNED_DATA,
)
from autosklearn.util.common import check_none
import numpy as np
import json
import sklearn
import pandas as pd
from pprint import pprint


def load_data(name):
    print(name)
    if name == "adult":
        X, y = sklearn.datasets.fetch_openml(
            data_id=1590, return_X_y=True, as_frame=True
        )
        y = (y == ">50K") * 1
        X = pd.get_dummies(X)
        X = X.drop("sex_Female", axis=1)
        X = X.rename(columns={"sex_Male": "sex"})
        return X, y
    if name == "compass":
        X, y = sklearn.datasets.fetch_openml(
            data_id=44162, return_X_y=True, as_frame=True
        )
        y = (y == "0") * 1
        X["race"] = (X["race"] == "0").astype(int)
        X = pd.get_dummies(X)
        return X, y
    if name == "lawschool":  # not finished
        X = pd.read_csv(
            "/work/dlclarge2/fetzert-MySpace/auto-sklearn/dataset/law_data.csv",
            dtype={"race": "category", "region_first": "category"},
            index_col=0,
        )
        X["sex"] = X["sex"].map({1: 0, 2: 1}).astype(int)
        # X = pd.get_dummies(X)
        y = X.pop("first_pf").apply(int).astype(int)
        # X["region_first"] = X["region_first"].astype(int)
        X = pd.get_dummies(X)
        X = X.drop(
            [
                "race_Amerindian",
                "race_Asian",
                "race_Black",
                "race_Hispanic",
                "race_Mexican",
                "race_Other",
                "race_Puertorican",
            ],
            axis=1,
        )
        X = X.rename(columns={"race_White": "race"})

        cols_to_convert = [
            "region_first_FW",
            "region_first_GL",
            "region_first_MS",
            "region_first_MW",
            "region_first_Mt",
            "region_first_NE",
            "region_first_NG",
            "region_first_NW",
            "region_first_PO",
            "region_first_SC",
            "region_first_SE",
            "race",
        ]

        # these is right now, if there issues with datatypes
        X[cols_to_convert] = X[cols_to_convert].astype(int)

        return X, y
    if name == "german":
        X, y = sklearn.datasets.fetch_openml(data_id=31, return_X_y=True, as_frame=True)
        y = (y == "good") * 1
        X["foreign_worker"] = (X["foreign_worker"] == "yes").astype(int)
        X = pd.get_dummies(X)
        return X, y
    if name == "crime":
        # not finished but prepared 
        X, y = sklearn.datasets.fetch_openml(data_id=315, return_X_y=True, as_frame=True)


    raise NotImplementedError


def save_pareto(
    automl,
    X_test,
    y_test,
    sensitive_features,
    runtime,
    fairness_constrain,
    methods,
    dataset,
    path
):
    pareto_front, scores = automl.get_pareto_set()
    # with open(path, "r") as f:
    #    infos = json.load(f)
    pprint(automl.cv_results_)
    """
    with open(path, "w") as f:
        test = []
        val = []
        for idx, ensemble in enumerate(pareto_front):
            predictions = ensemble.predict(X_test)
            acc = sklearn.metrics.accuracy_score(y_test, predictions)
            # TODO: needs adapted sometime
            fairness = fairlearn.metrics.demographic_parity_difference(
                y_test, predictions, sensitive_features=sensitive_features
            )
            test.append([1 - acc, fairness])
            val.append(list(scores[idx]))
        infos.append(
            {
                "dataset": dataset,
                "methods": methods,
                "runtime": runtime,
                "performance_metrics": "error",
                "fairness_metrics": fairness_constrain,
                "results": {"test": test, "val": val},
            }
        )
        #json.dump(infos, f)
    """

def demographic_parity_difference(solution, prediction, X_data, sensitive_features):
    sf = X_data[sensitive_features]
    return fairlearn.metrics.demographic_parity_difference(
        solution, prediction, sensitive_features=sf
    )


def equalized_odds_difference(solution, prediction, X_data, sensitive_features):
    sf = X_data[sensitive_features]
    return fairlearn.metrics.equalized_odds_difference(
        solution, prediction, sensitive_features=sf
    )


def consistency_score(solution, prediction, X_data):
    from aif360.sklearn.metrics import consistency_score

    return consistency_score(X_data, prediction)


def set_fair_metric(sf, metric):
    if metric == "demographic_parity":
        return autosklearn.metrics.make_scorer(
            name="demographic_parity_difference",
            score_func=demographic_parity_difference,
            optimum=0,
            greater_is_better=False,
            needs_proba=False,
            needs_X=True,
            needs_threshold=False,
            sensitive_features=sf,
        )
    if metric == "equalized_odds":
        return autosklearn.metrics.make_scorer(
            name="equalized_odds_difference",
            score_func=equalized_odds_difference,
            optimum=0,
            greater_is_better=False,
            needs_proba=False,
            needs_X=True,
            needs_threshold=False,
            sensitive_features=sf,
        )
    if metric == "consistency_score":
        return autosklearn.metrics.make_scorer(
            name="consistency_score",
            score_func=consistency_score,
            optimum=1,
            greater_is_better=True,
            needs_proba=False,
            needs_X=True,
            needs_threshold=False,
        )
    raise NotImplementedError


class LFR(AutoSklearnPreprocessingAlgorithm):
    index_sf = None
    # n_prototypes, reconstruct_weight, target_weight, fairness_weight, tol, max_iter,
    def __init__(
        self,
        n_prototypes=5,
        reconstruct_weight=0.01,
        target_weight=1,
        fairness_weight=50,
        tol=1e-4,
        max_iter=200,
        **kwargs,
    ):
        self.n_prototypes = n_prototypes
        self.reconstruct_weight = reconstruct_weight
        self.target_weight = target_weight
        self.fairness_weight = fairness_weight
        self.tol = tol
        self.max_iter = max_iter
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def set_fair_params(cls, index_sf):
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
            max_iter=self.max_iter,
        )
        # patched something in aif360, not good
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)#, self.preprocessor.predict(X)

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
        n_protoypes = UniformIntegerHyperparameter("n_protypes", 1, 10, default_value=5)
        reconstruct_weight = UniformFloatHyperparameter(
            "reconstruction_weight", 0.0001, 1, default_value=0.01, log=True
        )
        target_weight = UniformFloatHyperparameter(
            "target_weight", 0.01, 10, default_value=1, log=True
        )
        fairness_weight = UniformFloatHyperparameter(
            "fairness_weight", 0.5, 500, default_value=50, log=True
        )
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1, default_value=1e-4)
        max_iter = UniformIntegerHyperparameter("max_iter", 5000, 10000, default_value=5000)
        cs.add_hyperparameters(
            [
                n_protoypes,
                reconstruct_weight,
                target_weight,
                tol,
                fairness_weight,
                max_iter,
            ]
        )
        return cs


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
class SensitiveAtributeRemover(AutoSklearnPreprocessingAlgorithm):
    index_sf = None

    def __init__(self, random_state):
        """This preprocessors only remove the sensitive attributes"""

    @classmethod
    def set_fair_params(cls, index_sf):
        cls.index_sf = index_sf

    def fit(self, X, Y=None):
        self.preprocessor = "remover"
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        X.drop(columns=[SensitiveAtributeRemover.index_sf], inplace=True)
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



class CorrelationRemover(AutoSklearnPreprocessingAlgorithm):
    index_sf = []

    def __init__(self, alpha=1, **kwargs):
        self.alpha = alpha
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def set_fair_params(cls, index_sf):
        cls.index_sf.append(index_sf)

    def fit(self, X, y=None):
        from fairlearn.preprocessing import CorrelationRemover as FCR

        self.alpha = float(self.alpha)
        self.preprocessor = FCR(
            sensitive_feature_ids=CorrelationRemover.index_sf, alpha=self.alpha
        )

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
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


def set_fairlearn_attributes(index_sf, metric_name):

    classification.grid_search_decision_tree.GridSearch_DecisionTree.set_fair_params(
        index_sf, metric_name
    )
    classification.grid_search_adaboost.GridSearch_AdaboostClassifier.set_fair_params(
        index_sf, metric_name
    )
    classification.grid_search_bernoulli_nb.GridSearch_BernoulliNB.set_fair_params(
        index_sf, metric_name
    )
    classification.grid_search_gaussian_nb.GridSearch_GaussianNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.grid_search_lda.GridSearch_LDA.set_fair_params(index_sf, metric_name)
    classification.grid_search_liblinear_svc.GridSearch_LibLinear_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.grid_search_libsvm_svc.GridSearch_LibSVM_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.grid_search_multinomial_nb.GridSearch_MultinomialNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.grid_search_qda.GridSearch_QDA.set_fair_params(index_sf, metric_name)

    classification.exponentiated_gradient_decision_tree.ExponentiatedGradient_DecisionTree.set_fair_params(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_adaboost.ExponentiatedGradient_AdaboostClassifier.set_fair_params(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_bernoulli_nb.ExponentiatedGradient_BernoulliNB.set_fair_params(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_gaussian_nb.ExponentiatedGradient_GaussianNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.exponentiated_gradient_lda.ExponentiatedGradient_LDA.set_fair_params(index_sf, metric_name)
    classification.exponentiated_gradient_liblinear_svc.ExponentiatedGradient_LibLinear_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_libsvm_svc.ExponentiatedGradient_LibSVM_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_multinomial_nb.ExponentiatedGradient_MultinomialNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.exponentiated_gradient_qda.ExponentiatedGradient_QDA.set_fair_params(index_sf, metric_name)

    classification.threshold_optimizer_decision_tree.ThresholdOptimizer_DecisionTree.set_fair_params(
        index_sf, metric_name
    )
    classification.threshold_optimizer_adaboost.ThresholdOptimizer_AdaboostClassifier.set_fair_params(
        index_sf, metric_name
    )
    classification.threshold_optimizer_bernoulli_nb.ThresholdOptimizer_BernoulliNB.set_fair_params(
        index_sf, metric_name
    )
    classification.threshold_optimizer_gaussian_nb.ThresholdOptimizer_GaussianNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.threshold_optimizer_lda.ThresholdOptimizer_LDA.set_fair_params(index_sf, metric_name)
    classification.threshold_optimizer_liblinear_svc.ThresholdOptimizer_LibLinear_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.threshold_optimizer_libsvm_svc.ThresholdOptimizer_LibSVM_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.threshold_optimizer_multinomial_nb.ThresholdOptimizer_MultinomialNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.threshold_optimizer_qda.ThresholdOptimizer_QDA.set_fair_params(index_sf, metric_name)
    classification.threshold_optimizer_decision_tree.ThresholdOptimizer_DecisionTree.set_fair_params(
        index_sf, metric_name
    )

    classification.gs_threshold_optimizer_decision_tree.GS_ThresholdOptimizer_DecisionTree.set_fair_params(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_adaboost.GS_ThresholdOptimizer_AdaboostClassifier.set_fair_params(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_bernoulli_nb.GS_ThresholdOptimizer_BernoulliNB.set_fair_params(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_gaussian_nb.GS_ThresholdOptimizer_GaussianNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.gs_threshold_optimizer_lda.GS_ThresholdOptimizer_LDA.set_fair_params(index_sf, metric_name)
    classification.gs_threshold_optimizer_liblinear_svc.GS_ThresholdOptimizer_LibLinear_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_libsvm_svc.GS_ThresholdOptimizer_LibSVM_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_multinomial_nb.GS_ThresholdOptimizer_MultinomialNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.gs_threshold_optimizer_qda.GS_ThresholdOptimizer_QDA.set_fair_params(index_sf, metric_name)
    classification.gs_threshold_optimizer_decision_tree.GS_ThresholdOptimizer_DecisionTree.set_fair_params(
        index_sf, metric_name
    )

    classification.exg_threshold_optimizer_decision_tree.EXG_ThresholdOptimizer_DecisionTree.set_fair_params(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_adaboost.EXG_ThresholdOptimizer_AdaboostClassifier.set_fair_params(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_bernoulli_nb.EXG_ThresholdOptimizer_BernoulliNB.set_fair_params(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_gaussian_nb.EXG_ThresholdOptimizer_GaussianNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.exg_threshold_optimizer_lda.EXG_ThresholdOptimizer_LDA.set_fair_params(index_sf, metric_name)
    classification.exg_threshold_optimizer_liblinear_svc.EXG_ThresholdOptimizer_LibLinear_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_libsvm_svc.EXG_ThresholdOptimizer_LibSVM_SVC.set_fair_params(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_multinomial_nb.EXG_ThresholdOptimizer_MultinomialNB.set_fair_params(
        index_sf, metric_name
    )
    # classification.exg_threshold_optimizer_qda.EXG_ThresholdOptimizer_QDA.set_fair_params(index_sf, metric_name)
    classification.exg_threshold_optimizer_decision_tree.EXG_ThresholdOptimizer_DecisionTree.set_fair_params(
        index_sf, metric_name
    )


def add_correlation_remover(sf):
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(
        CorrelationRemover
    )
    CorrelationRemover.set_fair_params(sf)

def add_no_preprocessor():
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(
        no_preprocessor
    )



def add_sensitive_remover(index_sf):
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(
        SensitiveAtributeRemover
    )
    SensitiveAtributeRemover.set_fair_params(index_sf)


def add_LFR(index_sf):
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(LFR)
    LFR.set_fair_params(index_sf)
