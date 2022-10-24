#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b

import numpy as np
import argparse
import lzma
import sys
import pickle
import os
import urllib.request
import pandas as pd

from numpy.core.defchararray import isupper
from transformed_data import TransformedDataset
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(
        LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(
                url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Run prediction on given data")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument(
    "--window", default=4, type=int, help="Window size")


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = one_to_one_dict

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        test.data = " ".join(test.data.split())

        transformed_dataset = TransformedDataset(
            test, args.window, test.LETTERS_DIA + test.LETTERS_NODIA)

        with lzma.open(args.model_path, "rb") as model_file:
            pred_dict = pickle.load(model_file)
            model = pred_dict['MLP']
        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.

        new_text = list(test.data)

        diacritization = {"comma": {"a": "á", "n": "ń", "i": "í", "y": "ý", "e": "é", "o": "ó", "u": "ú"}, "hook": {
            "e": "ě", "s": "š", "c": "č", "r": "ř", "z": "ž", "n": "ň", "t": "ť", "d": "ď"}, "circle": {"u": "ů"}}

        predictions = np.argmax(model.predict_proba(
            transformed_dataset._data["windows"]), axis=1)

        def replace_using_dict(lowercase_data, dictionary):
            lowercase_wordwise = lowercase_data.split()
            lowercase_replaced = [
                dictionary[k] if k in dictionary else k for k in lowercase_wordwise]
            word_was_replaced = [
                1 if k in dictionary else 0 for k in lowercase_wordwise]
            new_data = lowercase_replaced
            # print(len(new_data), len(lowercase_wordwise))
            return new_data, word_was_replaced

        test_split, word_replacement_index = replace_using_dict(
            test.data, pred_dict['big_dict'])

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions_dict = ' '.join(test_split)

        for i in range(len(predictions)):
            if predictions[i] == 0:
                continue
            elif predictions[i] == 1 and str.lower(new_text[i]) in list(diacritization["comma"]):
                new_text[i] = str.upper(diacritization["comma"][str.lower(new_text[i])]) if new_text[i].isupper(
                ) else diacritization["comma"][str.lower(new_text[i])]
            elif predictions[i] == 2 and str.lower(new_text[i]) in list(diacritization["hook"]):
                new_text[i] = str.upper(diacritization["hook"][str.lower(new_text[i])]) if new_text[i].isupper(
                ) else diacritization["hook"][str.lower(new_text[i])]
            elif predictions[i] == 3 and str.lower(new_text[i]) in list(diacritization["circle"]):
                new_text[i] = str.upper(diacritization["circle"][str.lower(new_text[i])]) if new_text[i].isupper(
                ) else diacritization["circle"][str.lower(new_text[i])]

        predictions_model = "".join(new_text)

        dict_split = predictions_dict.split()
        model_split = predictions_model.split()
        final_split = []
        for index, word_indicator in enumerate(word_replacement_index):
            if word_indicator == 1:
                final_split.append(dict_split[index])
            else:
                final_split.append(model_split[index])
        output = " ".join(final_split)
        return output


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
