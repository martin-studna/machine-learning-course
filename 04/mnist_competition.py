#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np


# Use sklearn StandardScaler to scale pixel values
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import sklearn.model_selection
class Dataset:
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
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.2, type=float, help="Test size")
parser.add_argument("--final_train", default=True, type=bool, help="chomp up all training data")


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.

        # Create scale object
        #scaler = StandardScaler()
        # Fit scaler to training data only
        #scaler_fit = scaler.fit(X_train)
        # Transform both train and test data with the trained scaler
        #X_train_t = scaler_fit.transform(X_train)
        #X_test_t = scaler_fit.transform(X_test)

        if args.final_train:
            model = HistGradientBoostingClassifier(
                l2_regularization = 0.1,
                learning_rate=0.1,
                max_iter=200).fit(train.data, train.target)

            print("i got all the way over here", model.score(train.data, train.target))
        else:
            model = HistGradientBoostingClassifier(
                #max_depth=10,
                l2_regularization = 0.1,
                learning_rate=0.1,
                max_iter=200,
            verbose=1).fit(X_train, y_train)

            print("i got all the way over here", model.score(X_test, y_test))

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
