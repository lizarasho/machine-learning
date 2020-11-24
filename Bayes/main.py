import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt


def calc_p(count_x, counter, alpha, q=2):
    return (count_x + alpha) / (counter + alpha * q)


def map_gram(g):
    multiplier = 10 ** 5
    result = 0
    for i in range(len(g)):
        exp = multiplier ** (len(g) - 1 - i)
        result += g[i] * exp
    return result


def process_grams(a, n):
    indices = [i for i in range(len(a) + 1 - n)]
    grams = list(map(lambda ind: a[ind:(ind + n)], indices))
    return list(map(map_gram, grams))


def parse_file(filename, n):
    f = open(filename, 'r')
    subject = list(map(int, f.readline().split()[1:]))
    f.readline()
    message = list(map(int, f.readline().split()))
    return process_grams(subject, n) + process_grams(message, n)


def read_dataset(n, index):
    root = 'messages'
    dataset = []
    path = '{}/part{}'.format(root, index)
    for filename in glob.glob('{}/*.txt'.format(path)):
        c = int('spm' in filename)
        message = set(parse_file(filename, n))
        dataset.append((message, c))
    return dataset


def process_datasets(n, train_indices, test_index=None):
    train_dataset = []
    for i in train_indices:
        train_dataset += read_dataset(n, i)
    test_dataset = None
    if test_index is not None:
        test_dataset = read_dataset(n, test_index)
    return train_dataset, test_dataset


# logged_lambda = [1367, 0] — parameters to prevent legit messages from being classified as spam.
def bayes_classifier(train_dataset, test_dataset, alpha=1e-13, logged_lambdas=[0, 0]):
    num_classes = 2
    train_size = len(train_dataset)
    count = [{} for _ in range(num_classes)]
    classes_counter = [0 for _ in range(num_classes)]
    words_set = set()

    for i in range(train_size):
        message, c = train_dataset[i]
        classes_counter[c] += 1
        for word in message:
            words_set.add(word)
            if word not in count[c]:
                count[c][word] = 0
            count[c][word] += 1

    words_set = set(words_set)
    prior_prob = list(map(lambda x: x / train_size, classes_counter))
    p = [
        {w: np.log(calc_p(count[c].get(w, 0), classes_counter[c], alpha)) for w in words_set}
        for c in range(num_classes)
    ]
    rev_p = [
        {w: np.log(1 - calc_p(count[c].get(w, 0), classes_counter[c], alpha)) for w in words_set}
        for c in range(num_classes)
    ]
    precalced_sum = [np.sum([rev_p[c][w] for w in words_set]) for c in range(num_classes)]

    test_size = len(test_dataset)
    test_results = []
    for i in range(test_size):
        message, target = test_dataset[i]
        results = [logged_lambdas[c] + np.log(prior_prob[c]) + precalced_sum[c] for c in range(num_classes)]
        for c in range(num_classes):
            for w in message:
                results[c] -= rev_p[c].get(w, 0)
                results[c] += p[c].get(w, 0)
        predicted = np.argmax(results)
        normalized = np.divide(results, np.linalg.norm(results))
        test_results.append((predicted, target, normalized))
    return test_results


def calc_accuracy(results):
    true = 0
    for i in range(len(results)):
        predicted, target, _ = results[i]
        true += (predicted == target)
    return true / (len(results))


def draw_roc():
    n = 2
    parts_range = range(1, 11)

    precalced_datasets = [([], []) for _ in parts_range]
    for i in parts_range:
        train_range = [[j for j in parts_range if j != i]]
        precalced_datasets[i - 1] = process_datasets(n, train_range, i)
    all_results = []
    for i in parts_range:
        train_dataset, test_dataset = precalced_datasets[i - 1]
        test_results = bayes_classifier(train_dataset, test_dataset)
        all_results += test_results

    # Probability, predicted, target
    all_results = list(map(lambda a: (a[2][1], a[0], a[1]), all_results))
    all_results.sort(reverse=True)

    spams = len(list(filter(lambda a: a[2] == 1, all_results)))
    legits = len(all_results) - spams

    x, y = [0], [0]
    for _, predicted, target in all_results:
        if target == 1:
            x.append(x[-1])
            y.append(y[-1] + (1 / spams))
        else:
            x.append(x[-1] + (1 / legits))
            y.append(y[-1])

    plt.plot(x, y)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def precalc_datasets(n, parts_range):
    datasets = [([], []) for _ in parts_range]
    for i in parts_range:
        train_range = [[j for j in parts_range if j != i]]
        datasets[i - 1] = process_datasets(n, train_range, i)
    return datasets


def draw_plot():
    parts_range = range(1, 11)
    n = 2
    alpha = 1e-13
    lambdas = list(map(int, np.linspace(0, 1367, 100)))
    precalced_datasets = precalc_datasets(n, parts_range)

    x = lambdas
    y = []
    for l in lambdas:
        accuracies = []
        for i in parts_range:
            train_dataset, test_dataset = precalced_datasets[i - 1]
            test_results = bayes_classifier(train_dataset, test_dataset, alpha, logged_lambdas=[l, 0])
            accuracies.append(calc_accuracy(test_results))
        score = np.mean(accuracies)
        print('lambda = {}, score = {}'.format(l, score))
        y.append(score)
    plt.plot(x, y)
    plt.xlabel('ln(λ.legit)')
    plt.ylabel('accuracy')
    plt.show()


def validate():
    parts_range = range(1, 11)
    results = {'average score': [], 'n': [], 'alpha': []}
    n_grams = [1, 2, 3]
    alphas = [1e-1, 1e-2]
    precalced_datasets = [precalc_datasets(n - 1, parts_range) for n in n_grams]

    for n in n_grams:
        for alpha in alphas:
            accuracies = []
            for i in parts_range:
                train_dataset, test_dataset = precalced_datasets[n - 1][i - 1]
                test_results = bayes_classifier(train_dataset, test_dataset, alpha)
                accuracies.append(calc_accuracy(test_results))
            avg_score = np.mean(accuracies)
            results['average score'].append(avg_score)
            results['n'].append(n)
            results['alpha'].append('{:.0e}'.format(alpha))
            print('n = {}, alpha = {:.0e}, avg: {}'.format(n, alpha, avg_score))
    return pd.DataFrame.from_dict(results)


def main():
    table = validate()
    table = table.sort_values(by=['average score'], ascending=False)
    table.to_csv('results/alphas.csv')
    draw_roc()
    draw_plot()


if __name__ == '__main__':
    main()
