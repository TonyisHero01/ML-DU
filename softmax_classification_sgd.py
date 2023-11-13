#!/usr/bin/env python3

# fellas:
# 997bc6ac-2546-11ec-986f-f39926f24a9c
# 647545f8-2a7a-11ec-986f-f39926f24a9c
# 4820960d-2a7e-11ec-986f-f39926f24a9c

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def softmax(z, args):
    z -= np.max(z)

    denominator = 0
    for i in range(args.classes):
        denominator += np.exp(z[i])

    result = np.zeros(args.classes)
    for i in range(args.classes):
        result[i] = np.exp(z[i]) / denominator

    return result

def GetValueFromPredictions(p):
    currentMaxIndex = 0
    currentMaxIndexValue = p[0]

    for index in range(p.shape[0]):
        if (currentMaxIndexValue < p[index]):
            currentMaxIndex = index
            currentMaxIndexValue = p[index]

    return currentMaxIndex

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    ohe = sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    transformed_targets = ohe.fit_transform(train_target.reshape(-1, 1))

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.

        batch_permutations = np.array(np.split(permutation, len(permutation) / args.batch_size))

        for batch_permutation in batch_permutations:
            gradient = np.zeros((train_data.shape[1], args.classes))
            
            for index in batch_permutation:
                x = train_data[index] @ weights
                soft = softmax(x, args)
                gradient += np.outer(train_data[index], (soft - transformed_targets[index]))

            g = gradient / args.batch_size
            weights -= args.learning_rate * g

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.

        train_predictions = np.zeros((train_data.shape[0], args.classes))
        train_results = np.zeros(train_data.shape[0])
        for index in range(train_data.shape[0]):
            train_predictions[index] = softmax(train_data[index] @ weights, args)
        for index in range(train_results.shape[0]):
            train_results[index] = GetValueFromPredictions(train_predictions[index])

        test_predictions = np.zeros((test_data.shape[0], args.classes))
        test_results = np.zeros(test_data.shape[0])
        for index in range(test_data.shape[0]):
            test_predictions[index] = softmax(test_data[index] @ weights, args)
        for index in range(test_results.shape[0]):
            test_results[index] = GetValueFromPredictions(test_predictions[index])

        train_loss = sklearn.metrics.log_loss(ohe.fit_transform(train_target.reshape(-1, 1)), train_predictions)
        train_accuracy = sklearn.metrics.accuracy_score(train_target, train_results)

        test_loss = sklearn.metrics.log_loss(ohe.fit_transform(test_target.reshape(-1, 1)), test_predictions)
        test_accuracy = sklearn.metrics.accuracy_score(test_target, test_results)

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")