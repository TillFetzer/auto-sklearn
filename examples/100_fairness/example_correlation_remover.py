"""
Classification
==============

The following example shows how to fit *auto-sklearn* to optimize for two
competing metrics: `accuracy` and `demographic parity`.
This is an early example from the fairlearn preprocessing method correlation remover

If that not works, it should be reomeved before and can be removed

Auto-sklearn uses `SMAC3's implementation of ParEGO <https://automl.github.io/SMAC3/main/examples/3_multi_objective/2_parego.html#parego>`_.
Multi-objective ensembling and proper access to the full Pareto set will be added in the near
future.
"""
from typing import Optional
from pprint import pprint
import fairlearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
import autosklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.data_preprocessing
import sklearn.metrics


from ConfigSpace.configuration_space import ConfigurationSpace
from fairlearn.preprocessing import CorrelationRemover as FCR
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import (
    SPARSE,
    DENSE,
    UNSIGNED_DATA,
    INPUT,
    SIGNED_DATA,
)
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm

############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=1590, return_X_y=True, as_frame=True)
y = (y == ">50K") * 1
X = pd.get_dummies(X)
X = X.drop("sex_Female", axis=1)
X = X.rename(columns={"sex_Male": "sex"})


# Change the target to align with scikit-learn's convention that
# ``1`` is the minority class. In this example it is predicting
# that a credit is "bad", i.e. that it will default.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)
sf = "sex"
############################################################################
# Build and fit a classifier
# ==========================
# Add Correlation Remover component to auto-sklearn.
############################################################################
# Create Correlation Remover component for auto-sklearn
# =================================================


class CorrelationRemover(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, alpha=1, **kwargs):
        self.alpha = alpha
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fit(self, X, y=None):

        self.sensitive_ids = [sf]
        self.alpha = float(self.alpha)

        self.preprocessor = FCR(
            sensitive_feature_ids=self.sensitive_ids, alpha=self.alpha
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


# Add CorrelationRemover component to auto-sklearn.
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(CorrelationRemover)

############################################################################
# Configuration space
# ===================

cs = CorrelationRemover.get_hyperparameter_search_space()
print(cs)


# TODO include sensitive features in the fairlearn api
# time needs to be 1000 or more to see reasonable results
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=180,
    include={
        "data_preprocessor": ["CorrelationRemover"],
    },
    metric=[
        autosklearn.metrics.accuracy,
        autosklearn.metrics.demographic_parity_difference,
    ],
    initial_configurations_via_metalearning=0,
    # metric=autosklearn.metrics.accuracy,
    delete_tmp_folder_after_terminate=False,
)
autosklearn.metrics.set_sensitive_features(X_train, sf)
# X_train = X_train.drop(sf, axis=1)
# sensitive attributes needs to go out
automl.fit(X_train, y_train, dataset_name="adult")

############################################################################
# Compute the two competing metrics
# =================================
sensitive_features = X_test[sf]
# X_test = X_test.drop(sf, axis=1)

predictions = automl.predict(X_test)


print("Accuracy:", sklearn.metrics.accuracy_score(y_test, predictions))
print(
    "Fairness constrain",
    fairlearn.metrics.demographic_parity_difference(
        y_test, predictions, sensitive_features=sensitive_features
    ),
)
############################################################################
# View the models found by auto-sklearn
# =====================================
# They are by default sorted by the first metric given to *auto-sklearn*.

print(automl.leaderboard())

############################################################################
# ``cv_results`` also contains both metrics
# =========================================
# Similarly to the leaderboard, they are sorted by the first metric given
# to *auto-sklearn*.

pprint(automl.cv_results_)


############################################################################
# Visualize the Pareto set
# ==========================
plot_values = []
pareto_front = automl.get_pareto_set()
for ensemble in pareto_front:
    predictions = ensemble.predict(X_test)
    precision = sklearn.metrics.accuracy_score(y_test, predictions)
    recall = fairlearn.metrics.demographic_parity_difference(
        y_test, predictions, sensitive_features=sensitive_features
    )
    plot_values.append((precision, recall))
fig = plt.figure()
ax = fig.add_subplot(111)
for precision, recall in plot_values:
    ax.scatter(precision, recall, c="blue")
ax.set_xlabel("Accuracy")
ax.set_ylabel("fairness constrain")
ax.set_title("Pareto set")
plt.show()
