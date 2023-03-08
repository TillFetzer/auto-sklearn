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

# TODO change to new location if it is avaible
import set_fair_params

# logging.basicConfig(filename="/home/till/Documents/loogging.txt")

# TODO search if 5 is right now rigth


############################################################################
# Data Loading
# ============
def run_experiment(dataset, fairness_constrain, sf, runtime, file, seed, runcount):
    X, y = set_fair_params.load_data(dataset)
    # set_fair_params.set_fairlearn_attributes(X.columns.get_loc("sex"), "sex", "DemographicParity")
    # Change the target to align with scikit-learn's convention that
    # ``1`` is the minority class. In this example it is predicting
    # that a credit is "bad", i.e. that it will default.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1
    )
    # that can outsorced in another script.

    ############################################################################
    # Build and fit a classifier
    # ==========================

    fair_metric = set_fair_params.set_fair_metric(sf, fairness_constrain)
    set_fair_params.add_no_preprocessor()
    set_fair_params.add_correlation_remover(sf)

    ############################################################################
    # Build and fit a classifier
    # ==========================
    tmp =  file + "/{}/{}/{}/cr/{}times".format(fairness_constrain, dataset, seed, runcount)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=runtime,
        seed = seed,
        tmp_folder =  tmp + "/del", 
         # 3h
        metric=[
            autosklearn.metrics.accuracy,
            fair_metric,
        ],
        # metric=autosklearn.metrics.accuracy,
        delete_tmp_folder_after_terminate=False,
        initial_configurations_via_metalearning=0,
        smac_scenario_args={"runcount_limit": runcount},
        include={
            "data_preprocessor": ["CorrelationRemover"],
            "feature_preprocessor": ["no_preprocessing"], 
            "classifier": [
                "random_forest",
            ]
            # "GridSearch_DecisionTree", "ThresholdOptimizer_DecisionTree", , "decision_tree", "ExponentiatedGradient_DecisionTree"
        },
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
