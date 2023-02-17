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

# TODO change to new location if it is avaible
import set_fair_params

# logging.basicConfig(filename="/home/till/Documents/loogging.txt")

# TODO search if 5 is right now rigth


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=1590, return_X_y=True, as_frame=True)
y = (y == ">50K") * 1
X = pd.get_dummies(X)
X = X.drop("sex_Female", axis=1)
X = X.rename(columns={"sex_Male": "sex"})
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
sf = "sex"
fair_metric = set_fair_params.set_fair_metric(sf, "equalized_odds")
set_fair_params.add_correlation_remover(sf)


############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3 * 60 * 60,  # 3h
    metric=[
        autosklearn.metrics.accuracy,
        fair_metric,
    ],
    # metric=autosklearn.metrics.accuracy,
    delete_tmp_folder_after_terminate=False,
    initial_configurations_via_metalearning=0,
    include={
        "data_preprocessor": ["CorrelationRemover"],
        "classifier": [
            "adaboost",
            "bernoulli_nb",
            "decision_tree",
            "extra_trees",
            "gaussian_nb",
            "gradient_boosting",
            "k_nearest_neighbors",
            "lda",
            "liblinear_svc",
            "libsvm_svc",
            "mlp",
            "multinomial_nb",
            "passive_aggressive",
            "qda",
            "random_forest",
            "sgd",
        ]
        # "GridSearch_DecisionTree", "ThresholdOptimizer_DecisionTree", , "decision_tree", "ExponentiatedGradient_DecisionTree"
    },
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
    fairlearn.metrics.equalized_odds_difference(
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
result_values = []
pareto_front = automl.get_pareto_set()
with open("correlation_remover_demograpihcparity_long_eo.txt", "w") as f:
    for idx, ensemble in enumerate(pareto_front):
        predictions = ensemble.predict(X_test)
        acc = sklearn.metrics.accuracy_score(y_test, predictions)
        fairness = fairlearn.metrics.equalized_odds_difference(
            y_test, predictions, sensitive_features=sensitive_features
        )
        plot_values.append((acc, fairness))
        config = dict(ensemble.estimators[0].config)
        config["acc"] = acc
        config["fairnes"] = fairness
        f.write("Nummer {}: \n".format(idx + 1))
        f.writelines("{}:{}".format(line, config[line]) + "\n" for line in config)

fig = plt.figure()
ax = fig.add_subplot(111)
for precision, recall in plot_values:
    ax.scatter(precision, recall, c="blue")
ax.set_xlabel("Accuracy")
ax.set_ylabel("fairness constrain")
ax.set_title("Pareto set")
plt.show()
