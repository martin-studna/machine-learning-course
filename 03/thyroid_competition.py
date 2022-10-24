#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: IS (id yet unknown)
import argparse
import lzma
import os
import pickle
import urllib.request
from sklearn.preprocessing import PolynomialFeatures
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.decomposition
import sklearn.pipeline
import sklearn.preprocessing
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """

    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Run prediction on given data")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="thyroid_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.2,
                    type=float, help="relative test size")


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed)

        int_col_indices = np.where((np.mod(X_train, 1) != 0).sum(axis=0) == 0)[0]
        float_col_indices = np.where((np.mod(X_train, 1) != 0).sum(axis=0) != 0)[0]

        col_transformer = sklearn.compose.ColumnTransformer(
            [('encoder', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), int_col_indices),
             ("scaler", sklearn.preprocessing.StandardScaler(), float_col_indices)]
                                                            )

        pipeline = sklearn.pipeline.Pipeline([
            ("transformer", col_transformer),
            ("polynomial_features", sklearn.preprocessing.PolynomialFeatures()),
            ("logistic_regression", sklearn.linear_model.LogisticRegressionCV(

                Cs = np.geomspace(0.1, 100, 10) , solver = 'lbfgs', random_state=args.seed, max_iter=10000)),
        ])

        parameters = {"polynomial_features__degree": [1, 2], "logistic_regression__C": [
            0.01, 1, 2, 100], "logistic_regression__solver": ("lbfgs", "sag", "liblinear")}

        model = pipeline

        # TODO: Train a model on the given dataset and store it in `model`.
        model = model.fit(train.data, y=train.target)


        y_score = model.predict(X_test)
        y_score_proba = model.predict_proba(X_test)[:,1]
        scoring_df = pd.DataFrame({"y_score":y_score, "y_score_proba":y_score_proba, "y_test":y_test})
        scoring_df['quantile'] = pd.qcut(scoring_df["y_score_proba"], 20, labels=False)

        scoring_df['highperform_pred'] = model2.predict(data_transform_pipe.transform(X_test))

        scoring_df['final_score'] = np.where(scoring_df['quantile'] < 18, scoring_df['y_score'],
                                             scoring_df['highperform_pred']
                                             )
        scoring_df['final_score_hit'] = np.where(scoring_df['y_test'] == scoring_df['final_score'], 1,0)
        scoring_df['score_hit'] = np.where(scoring_df['y_test'] == scoring_df['y_score'], 1, 0)

        eval_table_quantiles = scoring_df.groupby('quantile')[['score_hit', "final_score_hit"]].mean()

        #print(eval_table_quantiles)
        print(scoring_df[['final_score_hit', 'score_hit']].mean()
              )



        scores = cross_val_score(model, train.data, train.target, cv=5)
        print("scores: ", scores,
              "model score on train: ", model.score(X_train, y=y_train),
              "model score on test: ", model.score(X_test, y=y_test),
        )

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
