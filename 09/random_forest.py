#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: Raphael Franke 346028f0-2825-11ec-986f-f39926f24a9c
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=True, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="digits", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=0.5, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=3, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=5, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

def criterion(train_target):
    values, counts = np.unique(train_target, return_counts=True)
    norm = len(train_target)
    p_T = np.array(counts) / norm
    c_T = -norm * np.sum([x * np.log10(x) for x in p_T if x != 0])
    return c_T

def node_split(node,args,generator):
    def subsample_features(number_of_features:int) -> np.ndarray:
        return generator.uniform(size=number_of_features) <= args.feature_subsampling
    sampled_features = subsample_features(node[0].shape[1])
    subsample = [i for i, x in enumerate(sampled_features) if x]
    node_data = node[0]
    node_target = node[1]
    best_points = np.zeros(node_data.shape[1])
    best_homogeneity = np.zeros(node_data.shape[1])
    node_L, node_R = [[], [], None], [[], [], None]
    for i in range(len(subsample)):
        options = np.unique(node_data[:, subsample[i]])
        node_data_ordered = node_data[np.argsort(node_data[:, subsample[i]])]
        node_target_ordered = node_target[np.argsort(node_data[:, subsample[i]])]
        if len(options) == 1:
            split_points = options
        else:
            split_points = [(options[j] + options[j + 1]) / 2 for j in range(len(options) - 1)]
        k = [len(np.where(node_data_ordered[:, subsample[i]] <= split_points[l])[0]) for l in range(len(split_points))]
        baseline = criterion(node_target_ordered)
        homogeneity = [
            criterion(node_target_ordered[0:k[j]]) + criterion(node_target_ordered[k[j]:]) - baseline
            for j in range(len(split_points))]
        best_points[subsample[i]] = split_points[np.argmin(homogeneity)]
        best_homogeneity[subsample[i]] = np.min(homogeneity)
    feat_split = np.argmin(best_homogeneity)  ##feature to split
    split_decision = best_points[feat_split]  ##split_point within feature
    for i in range(node_data.shape[0]):
        if node_data[i, feat_split] <= split_decision:
            node_L[0].append(node_data[i, :])
            node_L[1].append(node_target[i])
        else:
            node_R[0].append(node_data[i, :])
            node_R[1].append(node_target[i])
    pos_L = [node[2][0] + 1, 2 * node[2][1]]
    pos_R = [node[2][0] + 1, 2 * node[2][1] + 1]
    node_L = [np.array(node_L[0]), np.array(node_L[1]), pos_L]
    node_R = [np.array(node_R[0]), np.array(node_R[1]), pos_R]
    split = [feat_split, split_decision]
    return node_L, node_R, split, best_homogeneity[feat_split]

def decision_tree(node, args, res_split=[], pos=[], leaves=[], stem_pos=[],generator= None):
    leaves_pos, leaves_pred = [], []
    if pos == []: pos = [0, 0]
    ##max depth reached?
    if pos[0] < args.max_depth:
        node_L, node_R, split, homogeneity = node_split(node,args,generator)
        ##crit != 0?
        if homogeneity < 0:
            res_split.append(split)
            leaves.append(node_L)
            leaves.append(node_R)
            leaves = [leaves[i] for i in range(len(leaves)) if not np.array_equal(node[2], leaves[i][2])]
            stem_pos.append(pos)
            leaves, res_split, _, _, _, stem_pos = decision_tree(node_L, args, res_split, pos=node_L[2],
                                                                 leaves=leaves, stem_pos=stem_pos,generator = generator)
            leaves, res_split, _, _, _, stem_pos = decision_tree(node_R, args, res_split, pos=node_R[2],
                                                                 leaves=leaves, stem_pos=stem_pos, generator = generator)
            leaves_pos = [leaves[i][2] for i in range(len(leaves))]
            leaves_pred = [np.bincount(leaves[i][1]).argmax() for i in range(len(leaves))]
    return leaves, res_split, pos, leaves_pos, leaves_pred, stem_pos

def predict(data, pos, splits, leaves_pos, leaves_pred):
    i = 0
    predict = np.zeros(data.shape[0])
    for observation in data:
        obs_pos = [0,0]
        while obs_pos in pos:
            ind = pos.index(obs_pos)
            obs_pos[0] += 1
            if observation[splits[ind][0]] <= splits[ind][1]:
                obs_pos[1] = 2*obs_pos[1]
            else: obs_pos[1] = 2*obs_pos[1]+1
        ind = leaves_pos.index(obs_pos)
        predict[i] = leaves_pred[ind]
        i += 1
    return predict

def main(args: argparse.Namespace) -> tuple[float, float]:
    generator_feature_subsampling = np.random.RandomState(args.seed)
    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data:np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    # Create random generators
    predictions_train = np.zeros((args.trees,train_target.shape[0]),dtype = int)
    predictions_test = np.zeros((args.trees,test_target.shape[0]),dtype = int)
    for i in range(args.trees):
        if args.bagging:
            ind = bootstrap_dataset(train_data)
            root = [train_data[ind], train_target[ind], [0, 0]]
        else: root = [train_data,train_target, [0,0]]
        leaves, splits, pos, leaves_pos, leaves_pred, stem_pos = decision_tree(root, args, res_split=[], pos=[], leaves=[], stem_pos=[],generator = generator_feature_subsampling)
        predictions_train[i,:] = np.array(predict(train_data,stem_pos,splits, leaves_pos,leaves_pred),dtype = int)
        predictions_test[i,:] = np.array(predict(test_data, stem_pos, splits, leaves_pos, leaves_pred), dtype = int)
    test_pred = np.array([np.bincount(predictions_test[:,j]).argmax() for j in range(test_data.shape[0])])
    train_pred = np.array([np.bincount(predictions_train[:,j]).argmax() for j in range(train_data.shape[0])])

    # TODO: Create a random forest on the trainining data.

    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, so split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targets are not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in left subtrees before nodes in right subtrees.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating a feature mask using
    #     subsample_features(number_of_features)
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not
    #   (i.e., when feature_subsampling == 1, all features are used).
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with smallest class index in case of a tie.

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sum(train_pred == train_target)/len(train_data)
    test_accuracy = sum(test_pred == test_target)/len(test_data)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
