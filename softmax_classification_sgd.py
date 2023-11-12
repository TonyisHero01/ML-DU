#!/usr/bin/env python3
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
    #print(weights.shape)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        batch_permutations = np.array(np.split(permutation, len(permutation) / args.batch_size))

        for batch_permutation in batch_permutations:
            gradient = 0
            #print(batch_permutation.shape)
            
            for index in batch_permutation:
                #print("index: ")
                #print(index)
                y = train_data[index] @ weights
                
                for i in range(args.classes):
                    p = softmax(y)[i]
                    gradient += (p - train_target[index]) * train_data[index]
            g = gradient / args.batch_size
            weights[:,i] = weights[:,i] - args.learning_rate * g
        y = train_data[1] @ weights
        """
        print("y")
        print(y.shape)
        """
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        TW = train_data @ weights
        test_weights = test_data @ weights
        #train_predictions = np.ones((train_data.shape[0], args.classes))
        #test_predictions = np.ones((train_data.shape[0], args.classes))
        #print("test predict: ")
        #print(test_predictions.shape)
        """
        print("softmax")
        print(softmax(train_data @ weights).shape)
        """
        train_predictions = softmax(train_data @ weights)
        test_predictions = softmax(test_data @ weights)
        """
        print("test data")
        print(test_data.shape)
        print("weights")
        print(weights.shape)
        """
        train_loss = sklearn.metrics.log_loss(train_target, train_predictions)
        
        #print("funguje")
        test_loss = sklearn.metrics.log_loss(test_target, test_predictions)

        train_accuracies = np.ones(args.classes)
        test_accuracies = np.ones(args.classes)
        """
        print("train target")
        print(train_target.shape)
        print("TW")
        print(TW[:,1].shape)
        """
        for i in range(args.classes):
            train_accuracies[i] = sklearn.metrics.accuracy_score(train_target, TW[:,i] >= 0)
            test_accuracies[i] = sklearn.metrics.accuracy_score(test_target, test_weights[:,i] >= 0)
        train_accuracy = np.mean(train_accuracies)
        test_accuracy = np.mean(test_accuracies)
        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]

def softmax(z):
    # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
    max_ = np.max(z)
    z2 = z - max_
    res = np.ones(z2.shape)
    class_sum = 0
    for j in range(args.classes):
        class_sum += np.exp(z2[j])
    for i in range(z2.shape[0]):
        res[i] = z2[i]/class_sum
    return res


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
