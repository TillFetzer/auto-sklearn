from typing import Optional
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
import fairlearn.reductions
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from autosklearn.util.common import check_none
from pprint import pprint
import fairlearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
import autosklearn.metrics
import adding_fairlearn_methods

# logging.basicConfig(filename="/home/till/Documents/loogging.txt")

adding_fairlearn_methods.add_fairlearn_methods(5, "demographic_parity")


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=1590, return_X_y=True, as_frame=True)
y = (y == ">50K") * 1

# Change the target to align with scikit-learn's convention that
# ``1`` is the minority class. In this example it is predicting
# that a credit is "bad", i.e. that it will default.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)
# that can outsorced in another script.
def demographic_parity_difference(solution, prediction, X_data, sensitive_features):
    sf = X_data[sensitive_features]
    return fairlearn.metrics.demographic_parity_difference(
        solution, prediction, sensitive_features=sf
    )


############################################################################
# Build and fit a classifier
# ==========================
sf = "sex"
demographic_parity_difference = autosklearn.metrics.make_scorer(
    name="demographic_parity_difference",
    score_func=demographic_parity_difference,
    optimum=0,
    greater_is_better=False,
    needs_proba=False,
    needs_X=True,
    needs_threshold=False,
    sensitive_features=sf,
)
############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=10800,
    # per_run_time_limit=500,
    metric=[
        autosklearn.metrics.accuracy,
        demographic_parity_difference,
    ],
    # metric=autosklearn.metrics.accuracy,
    delete_tmp_folder_after_terminate=False,
    initial_configurations_via_metalearning=0,
    include={
        "feature_preprocessor": ["no_preprocessing"],
        "classifier": ["ThresholdOptimizer_DecisionTree"],
    },
    ensemble_class=None,
    ensemble_kwargs={"ensemble_size": 0},
)
# sensitive attributes needs to go out
automl.fit(X_train, y_train, dataset_name="adult")

############################################################################
# Compute the two competing metrics
# =================================
sensitive_features = X_test[sf]

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
