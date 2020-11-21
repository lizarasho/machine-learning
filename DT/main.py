import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


def calc_accuracy(tree, x_test, y_test):
    true = 0
    for i in range(len(x_test)):
        x = x_test[i]
        target = y_test[i]
        predicted = int(tree.predict([x]))
        true += (predicted == target)
    return true / len(x_test)


def calc_trees_accuracies(x_train, y_train, x_test, y_test):
    depths = [i for i in range(1, 101)]
    criteria = ['gini', 'entropy']
    splitters = ['best', 'random']
    results = {'accuracy': [], 'max_depth': [], 'criteria': [], 'splitter': []}
    for depth in depths:
        print('     depth = {}'.format(depth))
        for criterion in criteria:
            for splitter in splitters:
                tree = DecisionTreeClassifier(
                    max_depth=depth, criterion=criterion, splitter=splitter
                )
                tree.fit(x_train, y_train)
                accuracy = calc_accuracy(tree, x_test, y_test)
                results['accuracy'].append(accuracy)
                results['max_depth'].append(depth)
                results['criterion'].append(criterion)
                results['splitter'].append(splitter)
    return pd.DataFrame.from_dict(results)


def calc_forest_accuracy(source_train_dataset: pd.DataFrame, x_test, y_test):
    criterion = 'entropy'
    splitter = 'best'
    n = 100
    forest = []
    for i in range(n):
        train_dataset = source_train_dataset.sample(n=len(source_train_dataset.values), replace=True)
        x_train = np.delete(train_dataset.values, -1, axis=1)
        y_train = np.array(train_dataset.values[:, -1])
        tree = DecisionTreeClassifier(max_features=0.8, criterion=criterion, splitter=splitter)
        tree.fit(x_train, y_train)
        forest.append(tree)

    true = 0
    for i in range(len(x_test)):
        target = y_test[i]
        results = {}
        for tree in forest:
            predicted = int(tree.predict([x_test[i]]))
            if predicted not in results:
                results[predicted] = 0
            results[predicted] += 1
        predicted = sorted(results, key=results.get, reverse=True)[0]
        true += (predicted == target)
    accuracy = true / len(x_test)

    return accuracy


def read_dataset(filename):
    dataset = pd.read_csv(filename).values
    x_dataset = np.delete(dataset, -1, axis=1)
    y_dataset = np.array(dataset[:, -1])
    return x_dataset, y_dataset


def test_trees(tests):
    for num_test in tests:
        print('Run test {}'.format(num_test))
        x_train, y_train = read_dataset('DT_csv/{}_train.csv'.format(num_test))
        x_test, y_test = read_dataset('DT_csv/{}_test.csv'.format(num_test))
        table = calc_trees_accuracies(x_train, y_train, x_test, y_test)
        table = table.sort_values(by=['accuracy'], ascending=False)
        table.to_csv('results/{}.csv'.format(num_test))


def test_forest(tests):
    forest_results = {'n': [], 'accuracy': []}
    for num_test in tests:
        print('Run test {}'.format(num_test))
        train_dataset = pd.read_csv('DT_csv/{}_train.csv'.format(num_test))
        x_test, y_test = read_dataset('DT_csv/{}_test.csv'.format(num_test))
        forest_accuracy = calc_forest_accuracy(train_dataset, x_test, y_test)
        forest_results['n'].append(num_test)
        forest_results['accuracy'].append(forest_accuracy)
        print('test {}, accuracy = {}'.format(num_test, forest_accuracy))
    forest_table = pd.DataFrame.from_dict(forest_results)
    forest_table.to_csv('results/forest_results.csv')


def draw_plot(x_dataset, y_dataset, x_test, y_test, x_label):
    criterion = 'entropy'
    splitter = 'best'
    depths = [i for i in range(1, 101)]
    x = depths
    y = []
    for depth in depths:
        tree = DecisionTreeClassifier(
            max_depth=depth, criterion=criterion, splitter=splitter
        )
        tree.fit(x_dataset, y_dataset)
        accuracy = calc_accuracy(tree, x_test, y_test)
        y.append(accuracy)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel('accuracy')
    plt.show()


def draw_plots():
    x_train_03, y_train_03 = read_dataset('DT_csv/{}_train.csv'.format('03'))
    x_test_03, y_test_03 = read_dataset('DT_csv/{}_test.csv'.format('03'))
    x_train_12, y_train_12 = read_dataset('DT_csv/{}_train.csv'.format('12'))
    x_test_12, y_test_12 = read_dataset('DT_csv/{}_test.csv'.format('12'))
    draw_plot(x_train_03, y_train_03, x_test_03, y_test_03, '03, test dataset')
    draw_plot(x_train_03, y_train_03, x_train_03, y_train_03, '03, train dataset')
    draw_plot(x_train_12, y_train_12, x_test_12, y_test_12, '12, test dataset')
    draw_plot(x_train_12, y_train_12, x_train_12, y_train_12, '12, train dataset')


def main():
    tests = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
             '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    test_trees(tests)
    test_forest(tests)
    draw_plots()


if __name__ == '__main__':
    main()
