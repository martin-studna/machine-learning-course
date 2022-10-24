#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
import argparse
import os
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import pandas as pd


class MNIST:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """

    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--k", default=1, type=int,
                    help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int,
                    help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=1000,
                    type=int, help="Test set size")
parser.add_argument("--train_size", default=1000,
                    type=int, help="Train set size")
parser.add_argument("--weights", default="uniform", type=str,
                    help="Weighting to use (uniform/inverse/softmax)")
# If you add more arguments, ReCodEx will keep them with your default values.


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def main(args: argparse.Namespace) -> float:
    # Load MNIST data, scale it to [0, 1] and split it to train and test.
    mnist = MNIST(data_size=args.train_size + args.test_size)
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, test_size=args.test_size, random_state=args.seed)

    test_neighbors = np.zeros(test_data.shape[0])
    test_predictions = np.zeros(test_data.shape[0])

    for test_dato in range(test_data.shape[0]):
        dist_matrix = (
            np.sum(
                np.absolute(train_data - test_data[test_dato])
                ** args.p, axis=1
            ) ** (1 / args.p))

        first_k_sorted = np.argsort(dist_matrix)
        k_nearest = dist_matrix[first_k_sorted[:args.k]]
        k_nearest_indices = np.in1d(dist_matrix, k_nearest)

        k_nearest_distances = dist_matrix[k_nearest_indices]
        # X = np.reshape(dist_matrix,(1, dist_matrix.size))
        # Y =np.reshape(train_target,(1, train_target.size))
        # dist_matrix_targeted = np.concatenate((X, Y),axis = 0).shape

        # test_neighbors[test_dato] = k_nearest
        k_nearest_classes = train_target[k_nearest_indices]

        weighting_table = pd.DataFrame(
            {"distance": k_nearest_distances, "label": k_nearest_classes})

        if args.weights == "uniform":
            weighting_table['weight'] = 1
        if args.weights == "inverse":
            weighting_table['weight'] = weighting_table['distance'] ** (-1)
        if args.weights == "softmax":
            weighting_table['weight'] = softmax(-weighting_table['distance'])

        weighting_table_aggregated = (
            weighting_table
            .groupby("label").agg({"weight": "sum"})
            .sort_values(["weight", "label"], ascending=[False, True])
            .reset_index()
        )
        final_label = weighting_table_aggregated.iloc[0, 0]
        # unique, counts = np.unique(k_nearest_classes, return_counts=True)

        # freq_table =  pd.DataFrame(counts,unique)
        # sorted_freq_table = freq_table.reset_index().sort_values([0, "index"], ascending = [False, True])
        # final_label = sorted_freq_table.iloc[0,0]
        test_predictions[test_dato] = final_label

    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)
    accuracy

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                examples[test_target[i]] = [
                    test_data[i], *train_data[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example]
                    for example in examples if example]
        examples = [
            [example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1)
                   for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))
