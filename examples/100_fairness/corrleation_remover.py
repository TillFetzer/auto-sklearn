from ConfigSpace.configuration_space import ConfigurationSpace
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
import pandas as pd

# TODO change to new location if it is avaible
import set_fair_params

# logging.basicConfig(filename="/home/till/Documents/loogging.txt")




############################################################################
# Data Loading
# ============
def run_experiment(
    dataset, 
    fairness_constrain, 
    sf, 
    runtime, 
    file, 
    seed, 
    runcount=None, 
    under_folder="no_name", 
    configs = None):
    X, y = set_fair_params.load_data(dataset)
    # set_fair_params.set_fairlearn_attributes(X.columns.get_loc("sex"), "sex", "DemographicParity")
    # Change the target to align with scikit-learn's convention that
    # ``1`` is the minority class. In this example it is predicting
    # that a credit is "bad", i.e. that it will default.
    on = pd.concat([X[sf], y],axis=1)
    X_train , X_test, y_train, y_test, _, _ = set_fair_params.stratified_split(
        *(X.to_numpy(), y.to_numpy(), X[sf].to_numpy()), on = on ,size=0.8,  seed=seed
    )
    # that can outsorced in another script.
        # that can outsorced in another script.
    X_train, y_train = pd.DataFrame(X_train, columns=X.columns), pd.DataFrame(y_train)
    X_test, y_test = pd.DataFrame(X_test, columns=X.columns), pd.DataFrame(y_test)
    ############################################################################
    # Build and fit a classifier
    # ==========================
    if runcount:
        scenario_args = {"runcount_limit": runcount, "init_config": configs} if configs  else {"runcount_limit": runcount}
        #these only for structure in the folders
        runcount = "same_hyperparameter" if configs else str(runcount) + "strat"
        fair_metric = set_fair_params.set_fair_metric(sf, fairness_constrain)
        set_fair_params.add_no_preprocessor()
        set_fair_params.add_correlation_remover(sf)
    else: 
        runcount = runtime

    ############################################################################
    # Build and fit a classifier
    # ==========================train_ev
    tmp =  file + "/{}/{}/{}/{}/cr/{}".format(under_folder, fairness_constrain, dataset, seed, runcount)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=runtime,
        memory_limit=13000,
        seed = seed,
        tmp_folder =  tmp + "/del", 
         # 3h
        metric=[
            autosklearn.metrics.accuracy,
            fair_metric,
        ],
        # metric=autosklearn.metrics.accuracy,
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=15,
        smac_scenario_args=scenario_args,
        include={
            "feature_preprocessor": ["CorrelationRemover"],
            "data_preprocessor": ["no_preprocessor"], 
            "classifier": [
                "random_forest",
            ]
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
    automl.fit(X_train, y_train, dataset_name=dataset)

    ############################################################################
    # Compute the two competing metrics
    # =================================
    sensitive_features = X_test[sf]
    shutil.copy(tmp + "/del/smac3-output/run_{}/runhistory.json".format(seed), tmp )
    shutil.rmtree(tmp + "/del")
    print(
        "finished correlation remover {}s long on the {} dataset".format(
            runtime, dataset
        )
    )
