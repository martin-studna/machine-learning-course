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
parser.add_argument("--hidden_layer", default=50,
                    type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Learning rate")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(
        n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
        # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

        hidden_neurons = np.zeros((inputs.shape[0], args.hidden_layer))

        for i in range(inputs.shape[0]):
            hidden_neurons[i] = relu(
                inputs[i].T @ weights[0] + biases[0])

        output_neurons = np.zeros((inputs.shape[0], args.classes))

        for i in range(inputs.shape[0]):
            z = hidden_neurons[i].T @ weights[1] + biases[1]
            maximum_of_z = np.max(z)
            output_neurons[i] = softmax(z - maximum_of_z)

        return hidden_neurons, output_neurons

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of weights[0] and weights[1]
        # and gradient of biases[0] and biases[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to weights[1] and biases[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0] and biases[0]

        n_batches = int(train_data.shape[0] / args.batch_size)

        permuted_data = train_data[permutation]
        permuted_target = train_target[permutation]

        for i in range(n_batches):
            data_batch = permuted_data[i *
                                       args.batch_size:(i+1)*args.batch_size]
            target_batch = permuted_target[i *
                                           args.batch_size:(i+1)*args.batch_size]

            # Calculate the forward pass

            hidden_neurons, output_neurons = forward(data_batch)

            output_gradients = np.zeros(
                (data_batch.shape[0], args.classes))
            last_layer_gradients = np.zeros(
                (data_batch.shape[0], hidden_neurons.shape[1], output_gradients.shape[1]))
            hidden_layer_gradients = np.zeros(
                (data_batch.shape[0], hidden_neurons.shape[1]))
            input_gradients = np.zeros(
                (data_batch.shape[0], hidden_neurons.shape[1]))
            first_layer_gradients = np.zeros(
                (data_batch.shape[0], data_batch.shape[1], input_gradients.shape[1]))

            ones = np.zeros((data_batch.shape[0], args.classes))
            ones[np.arange(ones.shape[0]), target_batch] = 1

            for j in range(data_batch.shape[0]):
                output_gradients[j] = output_neurons[j] - ones[j]
                last_layer_gradients[j] = np.outer(
                    output_gradients[j], hidden_neurons[j]).T
                hidden_layer_gradients[j] = output_gradients[j] @ weights[1].T
                input_gradients[j] = hidden_layer_gradients[j] * \
                    (hidden_neurons[j] > 0)
                first_layer_gradients[j] = np.outer(
                    input_gradients[j], data_batch[j]).T

            output_gradient = np.mean(output_gradients, axis=0)
            last_layer_gradient = np.mean(last_layer_gradients, axis=0)
            last_layer_biases = np.mean(output_gradients, axis=0)
            first_layer_gradient = np.mean(first_layer_gradients, axis=0)
            first_layer_biases = np.mean(input_gradients, axis=0)
            weights[1] -= args.learning_rate * last_layer_gradient
            biases[1] -= args.learning_rate * last_layer_biases
            weights[0] -= args.learning_rate * first_layer_gradient
            biases[0] -= args.learning_rate * first_layer_biases

        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.
        train_accuracy, test_accuracy = 0, 0

        corr_pred = 0
        _, outputs = forward(train_data)
        y_preds = np.argmax(outputs, axis=1)
        corr_pred += np.sum(y_preds == train_target)
        train_accuracy = corr_pred / train_target.shape[0]

        corr_pred = 0
        _, outputs = forward(test_data)
        y_preds = np.argmax(outputs, axis=1)
        corr_pred += np.sum(y_preds == test_target)
        test_accuracy = corr_pred / test_target.shape[0]

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [train_accuracy, test_accuracy]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(
        w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")
