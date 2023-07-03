# this really should be implemented in other files, but right now, is here for testing
# and quick access

import autosklearn
import fairlearn

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.constants import (
    DENSE,
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

from examples.fairness.fairlearn_preprocessor import add_fair_preprocessor
# TODO better import
from examples.fairness.fairlearn_preprocessor.no_preprocessor import no_preprocessor
from examples.fairness.fairlearn_preprocessor.correlation_remover import CorrelationRemover
from examples.fairness.fairlearn_preprocessor.correlation_remover_dp import CorrelationRemover as CR
from examples.fairness.fairlearn_preprocessor.sensitive_attribute_remover import SensitiveAtributeRemover
from examples.fairness.fairlearn_preprocessor.learned_fair_representation import LFR
from examples.fairness.fairlearn_preprocessor.preferential_sampling import PreferentialSampling
from examples.fairness.fairlearn_preprocessor.no_fair_preprocessing import NoFairPreprocessor
# TODO: think on better options



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
        return X*1.0, y
    if name == "compass":
        X, y = sklearn.datasets.fetch_openml(
            data_id=44162, return_X_y=True, as_frame=True
        )
        y = (y == "0") * 1
        X["race"] = (X["race"] == "0").astype(int)
        X = pd.get_dummies(X)
        return X*1.0, y
    if name == "lawschool":  # not finished
        #todo datasets has to created
        X = pd.read_csv(
            "/work/ws/nemo/fr_tf167-conda-0/auto-sklearn/dataset/law_data.csv",
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
        X = X.rename(columns={"race_White": "race"})*1.0

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

        return X*1.0, y
    if name == "german":
        X, y = sklearn.datasets.fetch_openml(data_id=31, return_X_y=True, as_frame=True)
        y = (y == "good") * 1
        X["personal_status"] = X["personal_status"].str.contains("female").astype(int)
        X = pd.get_dummies(X)
        #X['foreign_worker_no'] = X['foreign_worker_no'].astype(int)
        #X['foreign_worker_yes'] = X['foreign_worker_yes'].astype(int)
        return X*1.0, y
    if name == "crime":
        # not finished but prepared 
        X, y = sklearn.datasets.fetch_openml(data_id=315, return_X_y=True, as_frame=True)


    raise NotImplementedError("{} dataset is not implemetened, maybe there is a typo".format(name))



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
        print(acc)
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
def f1_score(solution, prediction):
        return sklearn.metrics.f1_score(solution, prediction)


def set_f1_score():
        return autosklearn.metrics.make_scorer(
            name="f1",
            score_func=f1_score,
            optimum=1,
            greater_is_better=True,
            needs_proba=False,
            needs_X=False,
            needs_threshold=False
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
    from aif360.sklearn.metrics import equal_opportunity_difference
    sf = X_data[sensitive_features]
    return equal_opportunity_difference(
        solution, prediction, prot_attr=sensitive_features
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
def add_correlation_remover_dp(sf_index,sf):
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(
        CR
    )
    CR.utils_fairlearn(sf_index, sf)
def add_correlation_remover(sf):
    add_fair_preprocessor(CorrelationRemover)
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
    add_fair_preprocessor(LFR)
    LFR.utils_fairlearn(index_sf)

def add_preferential_sampling(index_sf):
    add_fair_preprocessor(PreferentialSampling)
    PreferentialSampling.utils_fairlearn(index_sf)
def add_no_fair():
    add_fair_preprocessor(NoFairPreprocessor)
   




#def add_preferential_sampling(index_sf):
#    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(PreferentialSampling)
#    PreferentialSampling.utils_fairlearn(index_sf)
