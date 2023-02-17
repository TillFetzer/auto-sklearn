import numpy as np

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics

"""
Finally, *Auto-sklearn* also support metric that require the train data (aka X_data) to
compute a value. This can be useful if one only cares about the score on a subset of the
data.
"""
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)


def accuracy(solution, prediction):
    # custom function defining accuracy
    return np.mean(solution == prediction)


def error(solution, prediction):
    # custom function defining error
    return np.mean(solution != prediction)


def accuracy_wk(solution, prediction, extra_argument):
    # custom function defining accuracy and accepting an additional argument
    assert extra_argument is None
    return np.mean(solution == prediction)


def error_wk(solution, prediction, extra_argument):
    # custom function defining error and accepting an additional argument
    assert extra_argument is None
    return np.mean(solution != prediction)


def metric_which_needs_x(solution, prediction, X_data, consider_col, val_threshold):
    # custom function defining accuracy
    assert X_data is not None
    rel_idx = X_data[:, consider_col] > val_threshold
    return np.mean(solution[rel_idx] == prediction[rel_idx])


accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu_X",
    score_func=metric_which_needs_x,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_X=True,
    needs_threshold=False,
    consider_col=1,
    val_threshold=18.8,
)
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    seed=1,
    metric=accuracy_scorer,
)
cls.fit(X_train, y_train)

predictions = cls.predict(X_test)
score = metric_which_needs_x(
    y_test,
    predictions,
    X_data=X_test,
    consider_col=1,
    val_threshold=18.8,
)
print(f"Error score {score:.3f} using {accuracy_scorer.name:s}")
