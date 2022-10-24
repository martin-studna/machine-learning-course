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
parser.add_argument("--C", default=1, type=float,
                    help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str,
                    help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int,
                    help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float,
                    help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int,
                    help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10,
                    type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-7, type=float,
                    help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    if args.kernel == "rbf":
        return np.exp(-args.kernel_gamma * np.linalg.norm(x-y, axis=-1)**2)
    elif args.kernel == "poly":
        return (args.kernel_gamma * np.dot(x, y) + 1) ** args.kernel_degree

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.


def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    passes_without_as_changing = 0
    train_accs, test_accs = [], []

    for _ in range(args.max_iterations):
        as_changed = 0
        E = np.zeros(len(a))

        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            E[i] = (np.sum(a * train_target * kernel(args,
                                                     train_data, train_data[i])) + b) - train_target[i]
            E[j] = (np.sum(a * train_target * kernel(args,
                                                     train_data, train_data[j])) + b) - train_target[j]

            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.

            if (a[i] < (args.C - args.tolerance) and train_target[i] * E[i] < - args.tolerance) or (a[i] > args.tolerance and train_target[i] * E[i] > args.tolerance):

                # If the conditions do not hold, then
                # - compute the updated unclipped a_j^new.
                #
                #   If the second derivative of the loss with respect to a[j]
                #   is > -`args.tolerance`, do not update a[j] and continue
                #   with next i.

                L_second_derivative = (2 * kernel(args, train_data[i], train_data[j]) - kernel(
                    args, train_data[i], train_data[i]) - kernel(args, train_data[j], train_data[j]))

                if L_second_derivative > - args.tolerance:
                    continue

                L_first_derivative = train_target[j] * (E[i] - E[j])

                new_a_j = a[j] - (L_first_derivative / L_second_derivative)

                # - clip the a_j^new to suitable [L, H].
                #
                #   If the clipped updated a_j^new differs from the original a[j]
                #   by less than `args.tolerance`, do not update a[j] and continue
                #   with next i.

                L, H = 0, 0
                if train_target[i] == train_target[j]:
                    L = max(0, a[i] + a[j] - args.C)
                    H = min(args.C, a[i] + a[j])
                else:
                    L = max(0, a[j] - a[i])
                    H = min(args.C, args.C + a[j] - a[i])

                new_a_j = np.clip(new_a_j, L, H)

                if abs(new_a_j - a[j]) < args.tolerance:
                    continue

                # - update a[j] to a_j^new, and compute the updated a[i] and b.
                #
                #   During the update of b, compare the a[i] and a[j] to zero by
                #   `> args.tolerance` and to C using `< args.C - args.tolerance`.

                new_a_i = a[i] - train_target[i] * \
                    train_target[j] * (new_a_j - a[j])

                new_b_j = b - E[j] - train_target[i] * (new_a_i - a[i]) * kernel(
                    args, train_data[i], train_data[j]) - train_target[j] * (new_a_j - a[j]) * kernel(args, train_data[j], train_data[j])

                new_b_i = b - E[i] - train_target[i] * (new_a_i - a[i]) * kernel(
                    args, train_data[i], train_data[i]) - train_target[j] * (new_a_j - a[j]) * kernel(args, train_data[j], train_data[i])

                new_b = 0
                if new_a_i > args.tolerance and new_a_i < (args.C - args.tolerance):
                    new_b = new_b_i
                elif new_a_j > args.tolerance and new_a_j < (args.C - args.tolerance):
                    new_b = new_b_j
                else:
                    new_b = (new_b_i + new_b_j) / 2

                a[i] = new_a_i
                a[j] = new_a_j
                b = new_b

                # - increase `as_changed`

                as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.

        corr_pred = 0
        for i in range(train_data.shape[0]):
            y_pred = np.sign(np.sum(a * train_target * kernel(args,
                                                              train_data, train_data[i])) + b)
            corr_pred += y_pred == train_target[i]

        train_acc = corr_pred / train_data.shape[0]

        corr_pred = 0
        for i in range(test_data.shape[0]):
            y_pred = np.sign(np.sum(a * train_target * kernel(args,
                                                              train_data, test_data[i])) + b)

            corr_pred += y_pred == test_target[i]

        test_acc = corr_pred / test_data.shape[0]

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).

    support_vectors = train_data[a > args.tolerance]
    support_vector_weights = a[a > args.tolerance] * \
        train_target[a > args.tolerance]

    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt

        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x)
                                    for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions,
                        levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target,
                        marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                        marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target,
                        marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1],
                        marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.
        def predict_function(x): return None

        plot(predict_function, support_vectors)
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
