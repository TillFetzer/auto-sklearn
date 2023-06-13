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
import utils_fairlearn
import json
from collections import defaultdict


def run_experiment(dataset, fairness_constrain, sf, runtime, file, seed, runcount, under_folder, performance = autosklearn.metrics.accuracy):
    X, y = utils_fairlearn.load_data(dataset)
   
    # ==========================
    on = pd.concat([X[sf], y],axis=1)
    X_train , X_test, y_train, y_test= utils_fairlearn.stratified_split(
        *(X.to_numpy(), y.to_numpy(), X[sf].to_numpy()),columns=X.columns,  on = on ,size=0.8,  seed=seed
    )

   
    ############################################################################
    # Build and fit a classifier
    # ==========================

    fair_metric = utils_fairlearn.set_fair_metric(sf, fairness_constrain)
  
    utils_fairlearn.add_no_preprocessor()

    ############################################################################
    # Build and fit a classifier
    # ==========================
    tmp =  file + "/{}/{}/{}/{}/so/{}timesstrat".format(under_folder, fairness_constrain, dataset, seed, runcount)
    runtime = runtime
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=runtime, 
        metric=performance,
        scoring_functions=[fair_metric],
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        smac_scenario_args={"runcount_limit": runcount},
        memory_limit=130000,
        seed = seed,
        tmp_folder =  tmp + "/del",
        include={
            'feature_preprocessor': ["no_preprocessing"],
            'data_preprocessor': ["no_preprocessor"],
            "classifier": [
                "random_forest"
            ], 
            # "GridSearch_DecisionTree", "ThresholdOptimizer_DecisionTree", , "decision_tree", "ExponentiatedGradient_DecisionTree"
        },
        resampling_strategy='fairness-cv',
        resampling_strategy_arguments= {
        "train_size": 0.67,     # The size of the training set
        "shuffle": True,        # Whether to shuffle before splitting data
        "folds": 5,             # Used in 'cv' based resampling strategies
        "groups": sf,
        "seed": seed             
        }
    )
    # sensitive attributes needs to go out
    automl.fit(X_train, y_train, dataset_name="adult")

    ############################################################################
    # Compute the two competing metrics
    # =================================
    #sensitive_features = X_test[sf]
   # utils_fairlearn.save_pareto(
    #    automl,
    #    X_test,
    #    y_test,
    #    sensitive_features,
    #    runtime,
    #    fairness_constrain,
    #    "moo",
    #    dataset,
    #    file
    #)
    
    fairness = automl.cv_results_["metric_{}".format(str(fair_metric))]
    with open(tmp + "/fairness.json", "w+") as f:
        fairness = json.dumps(list(fairness))
        f.write(fairness)
    shutil.copy(tmp + "/del/smac3-output/run_{}/runhistory.json".format(seed), tmp )
    shutil.rmtree(tmp + "/del")
   
