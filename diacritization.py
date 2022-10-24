#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn.preprocessing
from sklearn.neural_network import MLPClassifier
import sklearn.metrics

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

def features(x,size=5):
        features = [np.array(x[c-size:c+size+1]) for c in range(len(x))]
        for i in range(size):
            features[i] = np.array([1]*(size-i) + x[0:size+i+1])
            features[-i-1] = np.array(x[-(size+i)-1:] +[1]*(size-i))
        return features

def translate_from_dict(list,dictionary):
    for i in range(len(list)):
        val = list[i]
        if dictionary.get(val) == None:
            dictionary[val] = len(dictionary)+2
            list[i] = dictionary.get(val)
        else:
            list[i] = dictionary.get(val)
    return list, dictionary

def preprocessing(t_data,t_target,letters):
    data_temp = [c for c in t_data]
    target_temp = [c for c in t_target]
    alphabet = np.unique(target_temp)
    _, i = np.unique([ord(a) for a in alphabet], return_inverse=True)
    zip_iterator = zip(alphabet, i)
    dictionary = dict(zip_iterator)
    obs = [i for i in range(len(data_temp)) if data_temp[i] in letters]
    data_temp,_ = translate_from_dict(data_temp, dictionary)
    data = np.stack(features(data_temp))
    data = [data[i] for i in obs]
    target,_ = translate_from_dict(target_temp, dictionary)
    target = [target[i] for i in obs]
    return data,target,dictionary

def main(args: argparse.Namespace):
    letters= "acdeinorstuyzACDEINORSTUYZ"

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        data,target,dictionary = preprocessing(train.data,train.target,letters)
        # TODO: Train a model on the given dataset and store it in `model`.
        pipe = Pipeline([("onehot", sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore',sparse = True)),
                         ("mlp", MLPClassifier((500,300)))])
        model = pipe.fit(data,target)
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        with lzma.open('dictionary.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle)

    elif args.predict == "test":
        np.random.seed(args.seed)
        test = Dataset()
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with lzma.open('dictionary.pickle', 'rb') as handle:
            dictionary = pickle.load(handle)
        train_data, test_data, train_target, test_target = train_test_split(test.data, test.target, train_size=0.7,shuffle=False)
        data = test_data
        data_list = [c for c in data]
        data_temp = [c for c in data]
        obs = [i for i in range(len(data_temp)) if data_temp[i] in letters]
        data_trans,dictionary = translate_from_dict(data_temp, dictionary)
        data = np.stack(features(data_trans))
        data = [data[i] for i in obs]
        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        pred_temp = list(model.predict(data))
        inv_dictionary = {v: k for k, v in dictionary.items()}
        pred_temp, _ = translate_from_dict(pred_temp, inv_dictionary)
        prediction = data_list
        for i in range(len(obs)):
            prediction[obs[i]] = pred_temp[i]
        predictions = ''.join(prediction)
        test_target = ''.join(test_target)

    else:
        letters = "acdeinorstuyzACDEINORSTUYZ"
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with lzma.open('dictionary.pickle', 'rb') as handle:
            dictionary = pickle.load(handle)

        data = test.data
        data_list = [c for c in data]
        data_temp = [c for c in data]
        obs = [i for i in range(len(data_temp)) if data_temp[i] in letters]
        data_temp,dictionary = translate_from_dict(data_temp, dictionary)
        data = np.stack(features(data_temp))
        data = [data[i] for i in obs]
        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        pred_temp = list(model.predict(data))
        inv_dictionary = {v: k for k, v in dictionary.items()}
        pred_temp, _ = translate_from_dict(pred_temp, inv_dictionary)
        prediction = data_list
        for i in range(len(obs)):
            prediction[obs[i]] = pred_temp[i]
        predictions = ''.join(prediction)
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)