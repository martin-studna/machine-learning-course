#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os


import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier

class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx.".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")

parser.add_argument("--transformer_path", default="nli_competition.transformer", type=str, help="Model path")
parser.add_argument("--last_retrain", default=True, type=bool, help="gobble up all data for last retrain?")
parser.add_argument("--n_features", default=2 ** 16, type=int, help="number of features from hashing vectorizer")


parser.add_argument("--test_size", default=0.2, type=float, help="train-test split fraction")
#args = parser.parse_args([] if "__file__" not in globals() else None)


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8,
                                     # stop_words="english"
                                     )


        # TODO: Train a model on the given dataset and store it in `model`.

        all_data = vectorizer.fit_transform(train.data)
        # TODO: Train a model on the given dataset and store it in `model`.
        model = RidgeClassifier(tol=1e-2,  # solver="sag"
                                ).fit(all_data, train.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        with lzma.open(args.transformer_path, "wb") as transformer_file:
            pickle.dump(vectorizer, transformer_file)


    else:
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
