import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


def calc_accuracy(classifier, x_test, y_test):
    true = 0
    for i in range(len(x_test)):
        x = x_test[i]
        target = y_test[i]
        predicted = classifier(x)
        true += (predicted == target)
    return true / len(x_test)


def predict(tree, x):
    prediction = tree.predict([x])
    return np.sign(prediction[0])


def measure_quality(x_dataset, y_dataset, tree, weights):
    result = 0
    for i in range(len(x_dataset)):
        x, target = x_dataset[i], y_dataset[i]
        predicted = predict(tree, x)
        result += weights[i] * (target * predicted < 0)
    return result


def compose(a, b):
    def f(x):
        predicted = 0
        for t in range(len(a)):
            predicted += a[t] * predict(b[t], x)
        return np.sign(predicted)

    return f


def draw_plot(classifier, x_test, y_test, title):
    plt.title(title)
    min_x1, max_x1 = np.min(x_test[:, 0]), np.max(x_test[:, 0])
    min_x2, max_x2 = np.min(x_test[:, 1]), np.max(x_test[:, 1])
    dx1 = max_x1 - min_x1
    dx2 = max_x2 - min_x2
    step1 = 10
    step2 = 30
    for x1 in np.arange(min_x1 - dx1 / step1, max_x1 + dx1 / step1, dx1 / step2):
        for x2 in np.arange(min_x2 - dx2 / step1, max_x2 + step1 / step1, dx2 / step2):
            z = classifier([x1, x2])
            c = 'w'
            c = 'lightcoral' if z > 0 else c
            c = 'lightblue' if z < 0 else c
            plt.scatter(x1, x2, color=c, s=150, marker='s')
    for i in range(len(x_test)):
        x, y = x_test[i], y_test[i]
        c = 'b' if y == -1 else 'r'
        plt.scatter(x[0], x[1], color=c, s=10)
    plt.show()


def adaBoost(x_train, y_train, x_test, y_test, max_depth=1, T=55):
    objects_count = len(x_train)
    weights = [(1 / objects_count) for _ in range(objects_count)]
    a = [0.0 for _ in range(T)]
    b = [DecisionTreeClassifier(max_depth=max_depth) for _ in range(T)]
    steps = {1, 2, 3, 5, 8, 13, 21, 34, 55}
    results = []
    for t in range(T):
        tree = b[t]
        tree.fit(x_train, y_train, sample_weight=weights)
        Q = measure_quality(x_train, y_train, tree, weights)
        a[t] = np.log((1 - Q) / Q) / 2
        for i in range(len(weights)):
            weights[i] *= np.exp(-a[t] * y_train[i] * predict(tree, x_train[i]))
        weights_sum = np.sum(weights)
        for i in range(len(weights)):
            weights[i] /= weights_sum
        if t + 1 in steps:
            k = t + 1
            current_classifier = compose(a[:k], b[:k])
            score = calc_accuracy(current_classifier, x_test, y_test)
            draw_plot(current_classifier, x_test, y_test,
                      'max_depth = {}, step = {}, accuracy = {}'.format(max_depth, k, score))
            results.append((t + 1, score))
    return compose(a, b), results


def validate(x_dataset, y_dataset):
    max_depths = [1, 2, 3]
    validation_results = {'max_depth': [], 'step': [], 'accuracy': []}
    for depth in max_depths:
        print('depth = {}'.format(depth))
        _, boost_result = adaBoost(x_dataset, y_dataset, x_dataset, y_dataset, max_depth=depth)
        for step, score in boost_result:
            print('step = {}, accuracy = {}'.format(step, score))
            validation_results['max_depth'].append(depth)
            validation_results['step'].append(step)
            validation_results['accuracy'].append(score)
    table = pd.DataFrame.from_dict(validation_results)
    table.to_csv('results/chips_results.csv')


def kfold_validate(x_dataset, y_dataset):
    kf = KFold(n_splits=5, shuffle=True)
    max_depths = [1, 2, 3]
    validation_results = {'max_depth': [], 'step': [], 'accuracy': []}
    steps = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    ys = [[] for _ in range(len(max_depths))]
    for depth in max_depths:
        scores = {step: [] for step in steps}
        print('depth = {}'.format(depth))
        for train_index, test_index in kf.split(x_dataset):
            x_train, x_test = x_dataset[train_index], x_dataset[test_index]
            y_train, y_test = y_dataset[train_index], y_dataset[test_index]
            _, boost_result = adaBoost(x_train, y_train, x_test, y_test, max_depth=depth)
            for step, score in boost_result:
                scores[step].append(score)
        for step in steps:
            score = np.mean(scores[step])
            print('step = {}, accuracy = {}'.format(step, score))
            validation_results['max_depth'].append(depth)
            validation_results['step'].append(step)
            validation_results['accuracy'].append(score)
            ys[depth - 1].append(score)
    table = pd.DataFrame.from_dict(validation_results)
    table.to_csv('results/kfold_chips_results.csv')
    for depth in max_depths:
        plt.plot(steps, ys[depth - 1], label='max_depth = {}'.format(depth))
    plt.legend(loc='lower center')
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.show()


def read_dataset(filename):
    labels = {'P': 1., 'N': -1.}
    f = open(filename, 'r')
    f.readline()
    content = f.read().split()
    f.close()
    num_objects = len(content)
    x_dataset = [[] for _ in range(num_objects)]
    y_dataset = [0.0 for _ in range(num_objects)]
    for i in range(num_objects):
        line = content[i].split(',')
        x_dataset[i] = list(map(float, line[:-1]))
        y_dataset[i] = labels[line[-1]]
    return np.array(x_dataset), np.array(y_dataset)


def main():
    x_dataset, y_dataset = read_dataset('datasets/chips.csv')
    adaBoost(x_dataset, y_dataset, x_dataset, y_dataset, max_depth=2, T=55)
    validate(x_dataset, y_dataset)
    kfold_validate(x_dataset, y_dataset)


if __name__ == '__main__':
    main()
