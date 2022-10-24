#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int,
                    help="Number of SGD iterations over the data")
parser.add_argument("--l2", default=0.0, type=float,
                    help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_regression(
        n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = np.c_[data, np.ones(data.shape[0])]

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1])

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # A gradient for example (x_i, t_i) is `(x_i^T weights - t_i) * x_i`,
        # and the SGD update is
        #   weights = weights - args.learning_rate * (gradient + args.l2 * weights)`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        n_batches = int(train_data.shape[0] / args.batch_size)

        permuted_data = train_data[permutation]
        permuted_target = train_target[permutation]

        for i in range(n_batches):
            data_batch = permuted_data[i *
                                       args.batch_size:(i+1)*args.batch_size]
            target_batch = permuted_target[i *
                                           args.batch_size:(i+1)*args.batch_size]
            gradients = np.zeros((data_batch.shape[0], weights.shape[0]))
            for j in range(data_batch.shape[0]):
                gradients[j] = (data_batch[j].T @ weights -
                                target_batch[j]) * data_batch[j]

            gradient = np.mean(gradients, axis=0)

            weights = weights - args.learning_rate * \
                (gradient + args.l2 * weights)

        # TODO: Append current RMSE on train/test to train_rmses/test_rmses.
        train_rmses.append(np.sqrt(sklearn.metrics.mean_squared_error(
            train_target, np.dot(train_data, weights))))
        test_rmses.append(np.sqrt(sklearn.metrics.mean_squared_error(
            test_target, np.dot(test_data, weights))))

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on train_data (ignoring args.l2).
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(train_data, train_target)
    explicit_rmse = np.sqrt(sklearn.metrics.mean_squared_error(
        test_target, reg.predict(test_data)))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(
        sgd_rmse, explicit_rmse))
