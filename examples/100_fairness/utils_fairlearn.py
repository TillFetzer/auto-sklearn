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
    Constant
    
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

from sklearn.model_selection import train_test_split
from ConfigSpace.configuration_space import Configuration
def stratified_split(
    *x, on, size,columns, seed=1, ignore_missing=False
):
    # First we get the groups in the stratification, we need to make sure each of them exists
    # in the output
    stratification_groups = on.value_counts()
    min_name, min_val = min(stratification_groups.items(), key=lambda t: t[1])
    required_samples = len(x) / min_val

    if min_val == 1:
        raise RuntimeError(
            f"Can't create a stratified split with only 1 value for {min_name}"
            f"\n{stratification_groups}"
        )
    elif size < required_samples:
        raise RuntimeError(
            f"Sampling {size} but need at least {required_samples} samples"
            f" to accomodate lowest stratification group {min_name} with only"
            f" {min_val} samples"
        )

    # We split everything but also make sure to split the stratification column so
    # we can validate it
    X_train , X_test, y_train, y_test, _, _,sright, sleft = train_test_split(
        *x,
        on,
        train_size=size,
        stratify=on,
        random_state=seed,
    )

    if set(sright) != set(sleft) and not ignore_missing:
        raise RuntimeError(
            "Unique values to stratify on are not present in both splits"
            f" ,try increasing the split size"
            f"\nbefore {sright.value_counts()}"
            f"\nafter {sleft.value_counts()}"
        )
    # that can outsorced in another script.
    X_train, y_train = pd.DataFrame(X_train, columns=columns), pd.DataFrame(y_train)
    X_test, y_test = pd.DataFrame(X_test, columns=columns), pd.DataFrame(y_test)
    return X_train , X_test, y_train, y_test

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
            #"/home/till/Documents/auto-sklearn/dataset/law_data.csv",
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
        X["personal_status"] = X["personal_status"].str.contains("female").astype(int)
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
    
   # with open(path, "w") as f:
    test = []
    val = []
    for idx, ensemble in enumerate(pareto_front):
        predictions = ensemble.predict(X_test)
        acc = sklearn.metrics.accuracy_score(y_test, predictions)
            # TODO: needs adapted sometime
        fairness = fairlearn.metrics.equalized_odds_difference(
            y_test, predictions, sensitive_features=sensitive_features
        )
        test.append([1 - acc, fairness])
        val.append(list(scores[idx]))
        #infos.append(
        #    {
        #        "dataset": dataset,
        #        "methods": methods,
        #        "runtime": runtime,
        #        "performance_metrics": "error",
        #        "fairness_metrics": fairness_constrain,
        #        "results": {"test": test, "val": val},
        #    }
        #)
    print(test)
        #json.dump(infos, f)
    #"""

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

def equal_opportunity_difference(solution, prediction, X_data, sensitive_features):
    sf = X_data[sensitive_features]
    return aif360.sklearn.metrics.equal_opportunity_difference(
        solution, prediction, sensitive_features=sf
    )

def error_rate_difference(solution, prediction, X_data, sensitive_features):
    sf = X_data[sensitive_features] 
    sf = sf.reset_index().drop(columns='index')
   
    return abs(
        (1 - sklearn.metrics.accuracy_score(solution[(sf[sf[sensitive_features] == 0]).index.tolist()], prediction[(sf[sf[sensitive_features] == 0]).index.tolist()]))
    -   (1 - sklearn.metrics.accuracy_score(solution[(sf[sf[sensitive_features] == 1]).index.tolist()], prediction[(sf[sf[sensitive_features] == 1]).index.tolist()]))
    )

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
    if metric == "equal_opportunity_difference":
        return autosklearn.metrics.make_scorer(
            name="equal_opportunity_difference",
            score_func=equal_opportunity_difference,
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
    if metric == "error_rate_difference":
        return autosklearn.metrics.make_scorer(
                name="error_rate_difference",
                score_func=error_rate_difference,
                optimum=0,
                greater_is_better=False,
                needs_proba=False,
                needs_X=True,
                needs_threshold=False,
                sensitive_features=sf,
            )
    raise NotImplementedError


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
        predict_y = CategoricalHyperparameter("predict_y", [True, False])
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
    def utils_fairlearn(cls, index_sf):
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

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def utils_fairlearn(cls, index_sf):
        cls.index_sf.append(index_sf)

    def fit(self, X, y=None):
        from fairlearn.preprocessing import CorrelationRemover as FCR
        X = pd.DataFrame(X)
        self.alpha = float(self.alpha)
        self.preprocessor = FCR(
            sensitive_feature_ids=CorrelationRemover.index_sf, alpha=self.alpha
        )

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
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


def set_fairlearn_attributes(index_sf, metric_name):

    classification.grid_search_decision_tree.GridSearch_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )
    classification.grid_search_adaboost.GridSearch_AdaboostClassifier.utils_fairlearn(
        index_sf, metric_name
    )
    classification.grid_search_bernoulli_nb.GridSearch_BernoulliNB.utils_fairlearn(
        index_sf, metric_name
    )
    classification.grid_search_gaussian_nb.GridSearch_GaussianNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.grid_search_lda.GridSearch_LDA.utils_fairlearn(index_sf, metric_name)
    classification.grid_search_liblinear_svc.GridSearch_LibLinear_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.grid_search_libsvm_svc.GridSearch_LibSVM_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.grid_search_multinomial_nb.GridSearch_MultinomialNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.grid_search_qda.GridSearch_QDA.utils_fairlearn(index_sf, metric_name)

    classification.exponentiated_gradient_decision_tree.ExponentiatedGradient_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_adaboost.ExponentiatedGradient_AdaboostClassifier.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_bernoulli_nb.ExponentiatedGradient_BernoulliNB.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_gaussian_nb.ExponentiatedGradient_GaussianNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.exponentiated_gradient_lda.ExponentiatedGradient_LDA.utils_fairlearn(index_sf, metric_name)
    classification.exponentiated_gradient_liblinear_svc.ExponentiatedGradient_LibLinear_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_libsvm_svc.ExponentiatedGradient_LibSVM_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exponentiated_gradient_multinomial_nb.ExponentiatedGradient_MultinomialNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.exponentiated_gradient_qda.ExponentiatedGradient_QDA.utils_fairlearn(index_sf, metric_name)

    classification.threshold_optimizer_decision_tree.ThresholdOptimizer_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )
    classification.threshold_optimizer_adaboost.ThresholdOptimizer_AdaboostClassifier.utils_fairlearn(
        index_sf, metric_name
    )
    classification.threshold_optimizer_bernoulli_nb.ThresholdOptimizer_BernoulliNB.utils_fairlearn(
        index_sf, metric_name
    )
    classification.threshold_optimizer_gaussian_nb.ThresholdOptimizer_GaussianNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.threshold_optimizer_lda.ThresholdOptimizer_LDA.utils_fairlearn(index_sf, metric_name)
    classification.threshold_optimizer_liblinear_svc.ThresholdOptimizer_LibLinear_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.threshold_optimizer_libsvm_svc.ThresholdOptimizer_LibSVM_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.threshold_optimizer_multinomial_nb.ThresholdOptimizer_MultinomialNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.threshold_optimizer_qda.ThresholdOptimizer_QDA.utils_fairlearn(index_sf, metric_name)
    classification.threshold_optimizer_decision_tree.ThresholdOptimizer_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )

    classification.gs_threshold_optimizer_decision_tree.GS_ThresholdOptimizer_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_adaboost.GS_ThresholdOptimizer_AdaboostClassifier.utils_fairlearn(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_bernoulli_nb.GS_ThresholdOptimizer_BernoulliNB.utils_fairlearn(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_gaussian_nb.GS_ThresholdOptimizer_GaussianNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.gs_threshold_optimizer_lda.GS_ThresholdOptimizer_LDA.utils_fairlearn(index_sf, metric_name)
    classification.gs_threshold_optimizer_liblinear_svc.GS_ThresholdOptimizer_LibLinear_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_libsvm_svc.GS_ThresholdOptimizer_LibSVM_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.gs_threshold_optimizer_multinomial_nb.GS_ThresholdOptimizer_MultinomialNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.gs_threshold_optimizer_qda.GS_ThresholdOptimizer_QDA.utils_fairlearn(index_sf, metric_name)
    classification.gs_threshold_optimizer_decision_tree.GS_ThresholdOptimizer_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )

    classification.exg_threshold_optimizer_decision_tree.EXG_ThresholdOptimizer_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_adaboost.EXG_ThresholdOptimizer_AdaboostClassifier.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_bernoulli_nb.EXG_ThresholdOptimizer_BernoulliNB.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_gaussian_nb.EXG_ThresholdOptimizer_GaussianNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.exg_threshold_optimizer_lda.EXG_ThresholdOptimizer_LDA.utils_fairlearn(index_sf, metric_name)
    classification.exg_threshold_optimizer_liblinear_svc.EXG_ThresholdOptimizer_LibLinear_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_libsvm_svc.EXG_ThresholdOptimizer_LibSVM_SVC.utils_fairlearn(
        index_sf, metric_name
    )
    classification.exg_threshold_optimizer_multinomial_nb.EXG_ThresholdOptimizer_MultinomialNB.utils_fairlearn(
        index_sf, metric_name
    )
    # classification.exg_threshold_optimizer_qda.EXG_ThresholdOptimizer_QDA.utils_fairlearn(index_sf, metric_name)
    classification.exg_threshold_optimizer_decision_tree.EXG_ThresholdOptimizer_DecisionTree.utils_fairlearn(
        index_sf, metric_name
    )


def add_correlation_remover(sf):
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(
        CorrelationRemover
    )
    CorrelationRemover.utils_fairlearn(sf)

def add_no_preprocessor():
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(
        no_preprocessor
    )



def add_sensitive_remover(index_sf):
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(
        SensitiveAtributeRemover
    )
    SensitiveAtributeRemover.utils_fairlearn(index_sf)


def add_LFR(index_sf):
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(LFR)
    LFR.utils_fairlearn(index_sf)
