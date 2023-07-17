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


def run_experiment(dataset, fairness_constrain, sf, runtime, file, seed, runcount, under_folder, performance =  autosklearn.metrics.accuracy):
    X, y = utils_fairlearn.load_data(dataset)
    utils_fairlearn.add_no_preprocessor()
    utils_fairlearn.add_no_fair()
    utils_fairlearn.add_preferential_sampling(X.columns.get_loc(sf))
    utils_fairlearn.add_LFR(sf)
    utils_fairlearn.add_correlation_remover(sf)
    # ==========================
    on = pd.concat([X[sf], y],axis=1)
    X_train , X_test, y_train, y_test= utils_fairlearn.stratified_split(
        *(X.to_numpy(), y.to_numpy(), X[sf].to_numpy()),columns=X.columns,  on = on ,size=0.8,  seed=seed
    )

   
    ############################################################################
    # Build and fit a classifier
    # ==========================

    fair_metric = utils_fairlearn.set_fair_metric(sf, fairness_constrain)


    ############################################################################
    # Build and fit a classifier
    # ==========================
    tmp =  file + "/{}/{}/{}/{}/moo+ps+lfr/{}timesstrat".format(under_folder, fairness_constrain, dataset, seed, runcount)
    runtime = runtime
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=runtime,  # 3h

        #per_run_time_limit=runtime / 2,
        metric=[
           performance,
            fair_metric,
        ],
        # metric=autosklearn.metrics.accuracy,
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        smac_scenario_args={"runcount_limit": runcount},
        memory_limit=130000,
        seed = seed,
        tmp_folder =  tmp + "/del",
        include={
            'feature_preprocessor': ["no_preprocessing"],
            'data_preprocessor': ["no_preprocessor"],
            "fair_preprocessor": ["NoFairPreprocessor","PreferentialSampling", "LFR"],
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
    #cs = automl.get_configuration_space(X_train, y_train)
    #import pickle
    #with open("/home/till/Documents/auto-sklearn/tmp/configspace/moo+ps+lfr_config_space.pickle", "wb") as f:
    #    pickle.dump(cs, f)
    # sensitive attributes needs to go out
    automl.fit(X_train, y_train, dataset_name="adult")


    shutil.copy(tmp + "/del/smac3-output/run_{}/runhistory.json".format(seed), tmp )
    shutil.rmtree(tmp + "/del")
   
