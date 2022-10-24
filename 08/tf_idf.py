#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: Raphael Franke 346028f0-2825-11ec-986f-f39926f24a9c
import argparse
import lzma
import pickle
import os
import sys
import urllib.request
from scipy import sparse
import numpy as np
import re
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
from sklearn.preprocessing import normalize


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=True, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=37, type=int, help="Random seed")
parser.add_argument("--tf", default=True, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.

def main(args: argparse.Namespace) -> float:

    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)
    # TODO: Create a feature for every word that is present at least twice
    # in the training data. A word is every maximal sequence of at least 2 word characters,
    # where a word character corresponds to a regular expression `\w`.
    pattern = r'\w{2,}\b'
    ##find features
    docs = [re.findall(pattern, x) for x in train_data]
    flat_tokens = [item for sublist in docs for item in sublist]
    words, counts = np.unique(flat_tokens,return_counts = True)
    k = counts >= 2
    ind = np.where(k==1)[0]
    idx = {words[i]: [[0,0],[],[]] for i in ind}
    ##go through train_data
    id = 0
    for doc in docs:
        words, counts = np.unique(doc, return_counts=True)
        j = 0
        norm = sum(counts)
        for word in words:
            if word in idx:
                idx[word][1].append([id,(counts[j]/norm)])
                idx[word][0][0] += 1
            j += 1
        id += 1
    ##go through test_data
    docs = [re.findall(pattern, x) for x in test_data]
    id = 0
    for doc in docs:
        words, counts = np.unique(doc, return_counts=True)
        j = 0
        sum(counts)
        for word in words:
            if word in idx:
                idx[word][2].append([id, (counts[j] / norm)])
                idx[word][0][1] += 1
            j += 1
        id += 1
    train_matrix = sparse.lil_matrix((args.train_size, len(idx)))
    test_matrix = sparse.lil_matrix((args.test_size, len(idx)))
    # TODO: Weight the selected features using
    # - term frequency (TF), if `args.tf` is set;
    # - inverse document frequency (IDF), if `args.idf` is set; use
    #   the variant which contains `+1` in the denominator;
    # - TF * IDF, if both `args.tf` and `args.idf` are set;
    # - binary indicators, if neither `args.tf` nor `args.idf` are set.
    # Note that IDFs are computed on the train set and then reused without
    # modification on the test set, while TF is computed for every document separately.
    #
    # Finally, for each document L2-normalize its features.
    feat = 0
    for word in idx:
        if args.idf: idf = np.log10(args.train_size/(idx[word][0][0]+1))    ###maybe try other logarithms
        else: idf = 1
        if args.tf:
            for j in range(len(idx[word][1])):
                tf = idx[word][1][j][1]
                train_matrix[idx[word][1][j][0],feat] = tf*idf
            for j in range(len(idx[word][2])):
                tf = idx[word][2][j][1]
                test_matrix[idx[word][2][j][0],feat] = tf*idf
        else:
            for j in range(len(idx[word][1])):
                train_matrix[idx[word][1][j][0], feat] = idf
            for j in range(len(idx[word][2])):
                test_matrix[idx[word][2][j][0], feat] = idf
        feat += 1
    train_matrix = normalize(train_matrix,axis = 1)
    test_matrix = normalize(test_matrix, axis = 1)

    # TODO: Perform classification of the test set using the k-NN algorithm
    # from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
    # neighbors determined using the cosine similarity, where
    #   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
    # Note that for L2-normalized data (which we have), the nearest neighbors
    # are equivalent to using the usual Euclidean distance (L2 distance).
    kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors = args.k, algorithm = "brute", n_jobs = -1)
    kNN.fit(train_matrix,train_target)
    test_pred = kNN.predict(test_matrix)
    # TODO: Evaluate the performance using macro-averaged F1 score.

    f1_score = sklearn.metrics.f1_score(test_target,test_pred, average = "macro")
    return f1_score

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))
