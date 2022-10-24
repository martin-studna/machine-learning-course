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
import sklearn
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")
parser.add_argument("--transformer_path", default="isnt_it_ironic.transformer", type=str, help="Model path")
parser.add_argument("--last_retrain", default=True, type=bool, help="gobble up all data for last retrain?")
parser.add_argument("--n_features", default=2 ** 16, type=int, help="number of features from hashing vectorizer")


parser.add_argument("--test_size", default=0.2, type=float, help="train-test split fraction")
#args = parser.parse_args([] if "__file__" not in globals() else None)


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        X_train_list, X_test_list, y_train, y_test = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed)

        vectorizer = HashingVectorizer(
            stop_words="english", alternate_sign=False, n_features=args.n_features
        )
        X_train = vectorizer.transform(X_train_list)
        # TODO: Train a model on the given dataset and store it in `model`.
        model = NearestCentroid().fit(X_train, y_train)

        if args.last_retrain == True:

            X_train = vectorizer.transform(train.data)
            # TODO: Train a model on the given dataset and store it in `model`.
            model = NearestCentroid().fit(X_train, train.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        with lzma.open(args.transformer_path, "wb") as transformer_file:
            pickle.dump(vectorizer, transformer_file)


    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        with lzma.open(args.transformer_path, "rb") as transformer_file:
            transformer = pickle.load(transformer_file)

        # TODO: Generate `predictions` with the test set predictions.
        transformed_test_data = transformer.transform(test.data)
        predictions = model.predict(transformed_test_data)
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
