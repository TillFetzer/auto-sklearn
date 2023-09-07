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
import os
import tempfile

def run_experiment(dataset, fairness_constrain, sf, runtime, 
                   file, seed, runcount, under_folder,
                    performance =  autosklearn.metrics.accuracy, test=False):
    result_folder =  file + "/{}/{}/{}/{}/moo_ps_cr_lfr/{}timesstrat".format(under_folder, fairness_constrain, dataset, seed, runcount)
    runtime = runtime
    tempdir = tempfile.mkdtemp()
    autosklearn_directory = tempdir + 'dir_moo_ps_cr_lfr_{}'.format(seed)
    runhistory =  autosklearn_directory +  "/smac3-output/run_{}/runhistory.json".format(seed)
    if os.path.exists(result_folder):
        return
    X, y = utils_fairlearn.load_data(dataset)
    
    utils_fairlearn.add_correlation_remover(sf)
    # ==========================
    on = pd.concat([X[sf], y],axis=1)
    X_train , X_test, y_train, y_test= utils_fairlearn.stratified_split(
        *(X.to_numpy(), y.to_numpy(), X[sf].to_numpy()),columns=X.columns,  on = on ,size=0.8,  seed=seed
    )

   
    ############################################################################
    # Build and fit a classifier
    # ==========================
    sf = X.columns.get_loc(sf)
    fair_metric = utils_fairlearn.set_fair_metric(sf, fairness_constrain)
    X_train = pd.DataFrame(np.array(X_train))
    utils_fairlearn.add_no_fair()
    utils_fairlearn.add_no_preprocessor()
    #utils_fairlearn.add_correlation_remover(sf)
    utils_fairlearn.add_correlation_remover_dp(X.columns.get_loc(sf), sf)

    ############################################################################
    # Build and fit a classifier
    # ==========================
    tmp =  file + "/{}/{}/{}/{}/moo+cr/{}timesstrat".format(under_folder, fairness_constrain, dataset, seed, runcount)
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
            'feature_preprocessor': ["no_preprocessing","CorrelationRemover"],
            'data_preprocessor': ["no_preprocessor"],
            #"fair_preprocessor": ["NoFairPreprocessor","CorrelationRemover"],
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
    #with open("/home/till/Documents/auto-sklearn/tmp/moo_ps_cr_config_space.pickle", "wb") as f:
    #    pickle.dump(cs, f)
    # sensitive attributes needs to go out
    automl.fit(X_train, y_train, dataset_name="adult")

    ############################################################################
    # Compute the two competing metrics
    # =================================
    #sensitive_features = X_train[sf]
    #utils_fairlearn.save_pareto(
    #    automl,
    #    X_train,
    #    y_train,
    #    sensitive_features,
    #    runtime,
    #    fairness_constrain,
    #    "moo",
    #    dataset,
    #    file
    #)
    if test:
       utils_fairlearn.run_test_data(X_test, y_test, sf, fairness_constrain, automl, runhistory_file) 
    utils_fairlearn.save_history(autosklearn_directory, runhistory, result_folder)
    
    
   
