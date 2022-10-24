#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: Raphael Franke 346028f0-2825-11ec-986f-f39926f24a9c

import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import sklearn.compose
from sklearn.preprocessing import PolynomialFeatures


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.

parser.add_argument("--model_path", default="har.model", type=str, help="Model path")
parser.add_argument("--last_retrain", default=True, type=bool, help="gobble up all data for last retrain?")
parser.add_argument("--poly_degrees", default=2, type=int,
                    help="degree of polynomial features")

parser.add_argument("--test_size", default=0.2, type=float, help="train-test split fraction")
#args = parser.parse_args([] if "__file__" not in globals() else None)

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train_poly = sklearn.preprocessing.PolynomialFeatures(degree = args.poly_degrees).fit_transform(train.data)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.
        model = GradientBoostingClassifier(n_estimators=100,
                                             learning_rate=0.1,
                                             max_depth=10,
                                             random_state=0,
                                             verbose = True)

        if args.last_retrain == True:
            model = model.fit(train_poly, train.target)
        else:
            model = model.fit(X_train, y_train)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(
            sklearn.preprocessing.PolynomialFeatures(degree=args.poly_degrees).fit_transform(test.data)
        )

        return predictions

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
