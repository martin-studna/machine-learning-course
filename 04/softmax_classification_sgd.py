#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int,
                    help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int,
                    help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Learning rate")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

CONSTANT = 0.00001


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(
        n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = generator.uniform(
        size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        n_batches = int(train_data.shape[0] / args.batch_size)

        permuted_data = train_data[permutation]
        permuted_target = train_target[permutation]

        for i in range(n_batches):
            data_batch = permuted_data[i *
                                       args.batch_size:(i+1)*args.batch_size]
            target_batch = permuted_target[i *
                                           args.batch_size:(i+1)*args.batch_size]

            gradients = np.zeros(
                (data_batch.shape[0], weights.shape[0], args.classes))
            ones = np.zeros((data_batch.shape[0], args.classes))
            ones[np.arange(ones.shape[0]), target_batch] = 1

            for j in range(data_batch.shape[0]):
                z = data_batch[j].T @ weights
                maximum_of_z = np.max(z)
                gradients[j] = np.outer((softmax(z - maximum_of_z) -
                                         ones[j]), data_batch[j].T).T

            gradient = np.mean(gradients, axis=0)

            weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e.,i the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_accuracy, train_loss, test_accuracy, test_loss = 0, 0, 0, 0
        corr_pred = 0
        for i in range(train_data.shape[0]):
            train_loss += np.log(softmax(np.dot(train_data[i].T, weights)))[
                train_target[i]]
            y_pred = np.argmax(softmax(np.dot(train_data[i].T, weights)))
            corr_pred += y_pred == train_target[i]

        train_loss = - 1 * (train_loss / train_data.shape[0])
        train_accuracy = corr_pred / train_data.shape[0]

        corr_pred = 0
        for i in range(test_data.shape[0]):
            test_loss += np.log(softmax(np.dot(test_data[i].T, weights)))[
                test_target[i]]
            y_pred = np.argmax(softmax(test_data[i].T @ weights))
            corr_pred += y_pred == test_target[i]
        test_loss = - 1 * (test_loss / test_data.shape[0])
        test_accuracy = corr_pred / test_data.shape[0]

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, train_accuracy), (test_loss, test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w)
          for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
