#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: IS            2eff3afe-1393-11eb-8e81-005056ad4f31

import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn import linear_model


class Dataset:
    """Rental Dataset.
    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)
    The target variable is the number of rentals in the given hour.
    """

    def __init__(self,
                 name="rental_competition.train.npz",
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
    "--model_path", default="rental_competition.model", type=str, help="Model path")

parser.add_argument("--poly_degrees", default=2, type=int,
                    help="degree of polynomial features")
parser.add_argument("--test_size", default=0.2,
                    type=float, help="relative test size")
parser.add_argument("--final_retrain", default=True,
                    type=bool, help="should we split train/test or train on all we got")


def main(args: argparse.Namespace):
    poly = PolynomialFeatures(degree=args.poly_degrees, include_bias=False)
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        dataset = Dataset()

        # drop feeling temperature
        dataset.data = np.delete(dataset.data, 9, axis=1)
#        dataset.data = np.delete(dataset.data, 1, axis=1)  # drop year dummy

        int_col_indices = list(np.where(np.mod(dataset.data[0, :], 1) == 0)[0])
        float_col_indices = list(
            np.where(np.mod(dataset.data[0, :], 1) != 0)[0])

        col_transformer = sklearn.compose.ColumnTransformer(
            [('encoder', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), int_col_indices), ("scaler", sklearn.preprocessing.MinMaxScaler(), float_col_indices)])

        pipe = sklearn.pipeline.Pipeline([
            ('column_transformer', col_transformer),
            ('polynomial_transformer', poly)
        ])

        pipe.fit(dataset.data)
        transformed_data = pipe.transform(dataset.data)
        if args.final_retrain:
            X_train, y_train = transformed_data , dataset.target
            X_test, y_test = transformed_data, dataset.target

        else:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            transformed_data, dataset.target, test_size = args.test_size, random_state = args.seed)



            # TODO: Train a model on the given dataset and store it in `model`.
        ridge_cv_model = sklearn.linear_model.RidgeCV(
            alphas=np.logspace(-6, 6, 200)).fit(X_train, y_train)
        poisson_model = sklearn.linear_model.PoissonRegressor(
            alpha=0.9, max_iter=100000).fit(X_train, y_train)

        print('''training_set_score
         . 
        ''',
              "ridge_cv_model", ridge_cv_model.score(X_train, y_train),
              "poisson_model", poisson_model.score(X_train, y_train))

        print('''training_set_rmse
         . 
        ''',
              "ridge_cv_model", sklearn.metrics.mean_squared_error(
                  ridge_cv_model.predict(X_train), y_train, squared=False),

              "poisson_model", sklearn.metrics.mean_squared_error(
                  poisson_model.predict(X_train), y_train, squared=False),
              )

        # Serialize the model.

        print('''test_set_score
         . 
        ''',
              "ridge_cv_model", ridge_cv_model.score(X_test, y_test),
              "poisson_model", poisson_model.score(X_test, y_test))

        print('''test_set_rmse
         . 
        ''',
              "ridge_cv_model", sklearn.metrics.mean_squared_error(
                  ridge_cv_model.predict(X_test), y_test, squared=False),

              "poisson_model", sklearn.metrics.mean_squared_error(
                  poisson_model.predict(X_test), y_test, squared=False),
              )
        print("best ridge cv alpha",
              ridge_cv_model.best_score_, ridge_cv_model.alpha_)

        with lzma.open(f"{args.model_path}_pipe", "wb") as transform_file:
            pickle.dump(pipe, transform_file)

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(ridge_cv_model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)
        test.data = np.delete(test.data, 9, axis=1)
        with lzma.open("rental_competition.model", "rb") as model_file:
            model = pickle.load(model_file)

        with lzma.open(f"{args.model_path}_pipe", "rb") as transform_file:
            pipe = pickle.load(transform_file)

        test_transformed = pipe.transform(test.data)
        # TODO: Generate `predictions` with the test set predictions.

        predictions = model.predict(test_transformed)

        # rmse = sklearn.metrics.mean_squared_error(y_true, y_pred,
        # print rmse
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
