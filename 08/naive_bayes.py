#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: Raphael Franke 346028f0-2825-11ec-986f-f39926f24a9c
import argparse

import numpy as np
from scipy.stats import norm
from scipy.stats import norm
import math
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float,
                    help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="bernoulli",
                    type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int,
                    help="Number of classes")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(
        n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.

    _, class_patterns_count = np.unique(
        np.sort(train_target), return_counts=True)
    P_C = class_patterns_count / train_target.shape[0]

    test_accuracy = 0

    if args.naive_bayes_type == "gaussian":

        # Calculate means for every category and feature.
        means = np.array([np.mean(train_data[train_target == k], axis=0)
                          for k in range(args.classes)])

        # Compute variances for all each category and feature.
        variances = np.array([np.mean(
            (train_data[train_target == k] - means[k]) ** 2, axis=0) for k in range(args.classes)]) + args.alpha

        corr_pred = 0
        for i in range(test_data.shape[0]):
            probabilities = np.log(P_C) + np.sum(
                norm.logpdf(test_data[i], loc=means, scale=np.sqrt(variances)), axis=1)

            y_pred = np.argmax(probabilities)
            corr_pred += y_pred == test_target[i]

        test_accuracy = corr_pred / test_target.shape[0]

    if args.naive_bayes_type == "bernoulli":
        train_data = train_data >= 8
        test_data = test_data >= 8
        p_d_k = np.zeros((args.classes, train_data.shape[1]))
        for k in range(args.classes):
            data_class = train_data[train_target == k]
            p_d_k[k] = (np.sum(data_class, axis=0) + args.alpha) / \
                (data_class.shape[0] + 2 * args.alpha)

        corr_pred = 0
        for i in range(test_data.shape[0]):
            probabilities = np.log(
                P_C) + np.sum(np.log((p_d_k ** test_data[i]) * (1 - p_d_k) ** (1 - test_data[i])), axis=1)

            y_pred = np.argmax(probabilities)
            corr_pred += y_pred == test_target[i]

        test_accuracy = corr_pred / test_target.shape[0]

    if args.naive_bayes_type == "multinomial":

        p_d_k = np.zeros((args.classes, train_data.shape[1]))
        for k in range(args.classes):
            p_d_k[k] = ((np.sum(train_data[train_target == k],
                        axis=0) + args.alpha) / (np.sum(train_data[train_target == k]) + train_data.shape[1] * args.alpha))

        corr_pred = 0
        for i in range(test_data.shape[0]):
            probabilities = np.log(
                P_C) + np.sum(test_data[i] * np.log(p_d_k), axis=1)

            y_pred = np.argmax(probabilities)
            corr_pred += y_pred == test_target[i]

        test_accuracy = corr_pred / test_target.shape[0]

    # TODO: Predict the test data classes and compute test accuracy.

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
