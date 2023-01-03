"""
Classification
==============

The following example shows how to fit *auto-sklearn* to optimize for two
competing metrics: `accuracy` and `demographic parity`.

Auto-sklearn uses `SMAC3's implementation of ParEGO <https://automl.github.io/SMAC3/main/examples/3_multi_objective/2_parego.html#parego>`_.
Multi-objective ensembling and proper access to the full Pareto set will be added in the near
future.
"""
from pprint import pprint
import fairlearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
import autosklearn.metrics


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
sf = "sex"
############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=180,
    metric=[
        autosklearn.metrics.accuracy,
        autosklearn.metrics.demographic_parity_difference,
    ],
    # metric=autosklearn.metrics.accuracy,
    delete_tmp_folder_after_terminate=False,
)
autosklearn.metrics.set_sensitive_features(X_train, sf)
X_train = X_train.drop(sf, axis=1)
# sensitive attributes needs to go out
automl.fit(X_train, y_train, dataset_name="adult")

############################################################################
# Compute the two competing metrics
# =================================
sensitive_features = X_test[sf]
X_test = X_test.drop(sf, axis=1)

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
