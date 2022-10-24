#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
# TM3: Raphael Franke 346028f0-2825-11ec-986f-f39926f24a9c
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import time

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="digits", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default= None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=250, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def criterion(train_target, type):
    values, counts = np.unique(train_target,return_counts=True)
    norm = len(train_target)
    p_T = np.array(counts)/norm
    if type == "gini": c_T = norm*np.sum([x*(1-x) for x in p_T])
    elif type == "entropy": c_T = -norm*np.sum([x*np.log10(x) for x in p_T if x!= 0])
    return c_T


def node_split(node, type = "gini"):
    node_data = node[0]
    node_target = node[1]
    best_points = np.zeros(node_data.shape[1])
    best_homogeneity = np.zeros(node_data.shape[1])
    node_L, node_R = [[],[],None],[[],[],None]
    for i in range(node_data.shape[1]):
        options = np.unique(node_data[:,i])
        node_data_ordered = node_data[np.argsort(node_data[:,i])]
        node_target_ordered = node_target[np.argsort(node_data[:,i])]
        if len(options)==1: split_points = options
        else: split_points = [(options[j]+options[j+1])/2 for j in range(len(options)-1)]
        k = [len(np.where(node_data_ordered[:,i] <= split_points[l])[0]) for l in range(len(split_points))]
        baseline = criterion(node_target_ordered,type)
        homogeneity = [criterion(node_target_ordered[0:k[j]],type) + criterion(node_target_ordered[k[j]:],type) - baseline for j in range(len(split_points))]
        best_points[i] = split_points[np.argmin(homogeneity)]
        best_homogeneity[i] = np.min(homogeneity)
    feat_split = np.argmin(best_homogeneity) ##feature to split
    split_decision = best_points[feat_split] ##split_point within feature
    for i in range(node_data.shape[0]):
        if node_data[i,feat_split] <= split_decision:
            node_L[0].append(node_data[i,:])
            node_L[1].append(node_target[i])
        else:
            node_R[0].append(node_data[i,:])
            node_R[1].append(node_target[i])
    pos_L = [node[2][0]+1,2*node[2][1]]
    pos_R = [node[2][0]+1, 2*node[2][1]+1]
    node_L = [np.array(node_L[0]),np.array(node_L[1]),pos_L]
    node_R = [np.array(node_R[0]),np.array(node_R[1]),pos_R]
    split = [feat_split, split_decision]

    return node_L, node_R, split, best_homogeneity[feat_split]

def splitting(node, args, res_split = [], pos = [], leaves = [], stem_pos = [], type = "gini"):
    if args.max_leaves is None:
        leaves_pos, leaves_pred= [],[]
        if pos == []: pos = [0,0]
        if args.max_depth is None or pos[0] < args.max_depth:
            if args.min_to_split is None or len(node[1]) >= args.min_to_split:
                node_L,node_R, split, homogeneity = node_split(node,type)
                if homogeneity < 0:
                    res_split.append(split)
                    leaves.append(node_L)
                    leaves.append(node_R)
                    leaves = [leaves[i] for i in range(len(leaves)) if not np.array_equal(node[2],leaves[i][2])]
                    stem_pos.append(pos)
                    leaves, res_split, _, _, _,stem_pos = splitting(node_L, args, res_split, pos=node_L[2], leaves=leaves,stem_pos = stem_pos,type = type)
                    leaves, res_split, _, _, _, stem_pos = splitting(node_R, args, res_split, pos=node_R[2],leaves = leaves, stem_pos = stem_pos, type= type)
                    leaves_pos = [leaves[i][2] for i in range(len(leaves))]
                    leaves_pred = [np.bincount(leaves[i][1]).argmax() for i in range(len(leaves))]

    else:
        if leaves == []: leaves = [node]
        node_L_ref, node_R_ref, split_ref, homo_ref, leaves_pos, leaves_pred = [], [], [], None, [], []
        if len(leaves) < args.max_leaves:
            num  = 0
            for node in leaves:
                if args.min_to_split is None or len(node[1]) >= args.min_to_split:
                    node_L, node_R, split, homogen = node_split(node, type)
                    if homo_ref == None:
                        homo_ref = homogen
                        node_L_ref = node_L
                        node_R_ref = node_R
                        split_ref = split
                        num_ref = num
                    else:
                        if homogen < homo_ref:
                            homo_ref = homogen
                            node_L_ref = node_L
                            node_R_ref = node_R
                            split_ref = split
                            num_ref = num
                num += 1
            if homo_ref < 0:
                stem_pos.append(leaves[num_ref][2])
                del leaves[num_ref]
                leaves.append(node_L_ref)
                leaves.append(node_R_ref)
                res_split.append(split_ref)
                leaves, _, _ ,_,_,stem_pos = splitting(leaves,args, res_split,pos = pos, leaves= leaves, stem_pos = stem_pos, type = type)
                leaves_pos = [leaves[i][2] for i in range(len(leaves))]
                leaves_pred = [np.bincount(leaves[i][1]).argmax() for i in range(len(leaves))]
    return leaves, res_split, pos, leaves_pos,leaves_pred,stem_pos

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
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    root = [train_data,train_target,[0,0]]
    leaves, splits, pos, leaves_pos, leaves_pred,stem_pos = splitting(root,args, type = args.criterion)
    train_pred = np.array(predict(train_data,stem_pos,splits, leaves_pos,leaves_pred),dtype = int)
    print(splits)
    print(stem_pos)
    test_pred = np.array(predict(test_data,stem_pos, splits, leaves_pos,leaves_pred),dtype = int)
    train_accuracy = sum(train_pred == train_target)/len(train_data)
    test_accuracy = sum(test_pred == test_target)/len(test_data)

    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3 the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
