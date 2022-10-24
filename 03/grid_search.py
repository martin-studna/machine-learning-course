#!/usr/bin/env python3

# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: IS (id yet unknown)

import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(random_state=args.seed)
    #
    # Then, using sklearn.model_selection.StratifiedKFold(5), evaluate crossvalidated
    # train performance of all combinations of the the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    #
    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    pipeline = sklearn.pipeline.Pipeline([
        ("min_max_scaler", sklearn.preprocessing.MinMaxScaler()),
        ("polynomial_features", sklearn.preprocessing.PolynomialFeatures()),
        ("logistic_regression", sklearn.linear_model.LogisticRegression(
            random_state=args.seed)),
    ])

    parameters = {"polynomial_features__degree": [1, 2], "logistic_regression__C": [
        0.01, 1, 100], "logistic_regression__solver": ("lbfgs", "sag")}
    grid_search = sklearn.model_selection.GridSearchCV(pipeline, parameters)

    grid_search.fit(X_train, y=y_train)
    y_pred = grid_search.predict(X_test)

    test_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))
