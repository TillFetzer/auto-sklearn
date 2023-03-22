from typing import Optional
import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
import fairlearn.reductions
from autosklearn.askl_typing import FEAT_TYPE_TYPE
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE
from autosklearn.util.common import check_none
from pprint import pprint
import fairlearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
import autosklearn.metrics

import shutil

# TODO change to new location if it is avaible
import set_fair_params
import json
from collections import defaultdict


def run_experiment(dataset, fairness_constrain, sf, runtime, file, seed, runcount, under_folder):
    X, y = set_fair_params.load_data(dataset)

    # ==========================

    fair_metric = set_fair_params.set_fair_metric(sf, fairness_constrain)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1
    )

    # that can outsorced in another script.

    ############################################################################
    # Build and fit a classifier
    # ==========================

    fair_metric = set_fair_params.set_fair_metric(sf, fairness_constrain)
    set_fair_params.add_LFR(sf)
    set_fair_params.add_no_preprocessor()

    ############################################################################
    # Build and fit a classifier
    # ==========================
    tmp =  file + "/{}/{}/{}/{}/lfr/{}timesstr".format(under_folder, fairness_constrain, dataset, seed, runcount)
    runtime = runtime
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=runtime,  # 3h

        #per_run_time_limit=runtime / 2,
        metric=[
            autosklearn.metrics.accuracy,
            fair_metric,
        ],
        # metric=autosklearn.metrics.accuracy,
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        smac_scenario_args={"runcount_limit": runcount},
        memory_limit=6174,
        seed = seed,
        tmp_folder =  tmp + "/del",
        include={
            'feature_preprocessor': ["no_preprocessing"],
            'data_preprocessor': ["LFR"],
            "classifier": [
                "random_forest",
            ]
            # "GridSearch_DecisionTree", "ThresholdOptimizer_DecisionTree", , "decision_tree", "ExponentiatedGradient_DecisionTree"
        },
        resampling_strategy='fairness-holdout',
        resampling_strategy_arguments= {
        "train_size": 0.67,     # The size of the training set
        "shuffle": True,        # Whether to shuffle before splitting data
        "folds": 5,             # Used in 'cv' based resampling strategies
        "groups": sf,
        "seed": seed             
        }
    )
    # sensitive attributes needs to go out
    automl.fit(X_train, y_train)

    ############################################################################
    # Compute the two competing metrics
    # =================================
    sensitive_features = X_test[sf]
    #set_fair_params.save_pareto( automl,
    #X_test,
    #y_test,
    #sensitive_features,
    #runtime,
    #fairness_constrain,
    #"test",
    #dataset,
    #"/home/till/Documents/auto-sklearn/results_t.json")

    shutil.copy(tmp + "/del/smac3-output/run_{}/runhistory.json".format(seed), tmp )
    shutil.rmtree(tmp + "/del")
  