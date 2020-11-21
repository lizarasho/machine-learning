import numpy as np
import random
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import time


def polynomial_kernel(degree):
    def kernel(x, y):
        return np.dot(np.transpose(x), y) ** degree

    return kernel


linear_kernel = polynomial_kernel(1)


def gaussian_kernel(beta):
    def kernel(x, y):
        return np.exp(-beta * np.linalg.norm((x - y) ** 2))

    return kernel


def parse_object(i, dataset):
    return dataset[i][:-1], dataset[i][-1]


def calc_f(x, dataset, alphas, beta, K):
    return beta + np.sum([
        alphas[i] * dataset[i][-1] * K(dataset[i][:-1], x) for i in range(len(dataset))]
    )


def random_except(max_val, except_val):
    result = except_val
    while result == except_val:
        result = random.randint(0, max_val - 1)
    return result


def calc_limits(y_i, y_j, alpha_i, alpha_j, C):
    if y_i == y_j:
        return max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j)
    return max(0, alpha_j - alpha_i), min(C, C + alpha_j - alpha_i)


def update_beta(beta_1, beta_2, alpha_i, alpha_j, C):
    if 0 < alpha_i < C:
        return beta_1
    if 0 < alpha_j < C:
        return beta_2
    return (beta_1 + beta_2) / 2


def sequential_minimal_optimization(dataset: np.ndarray, kernel_function, C, passes_limit=25, tolerance=1e-7):
    num_objects, _ = dataset.shape
    alphas = np.zeros(num_objects)
    beta = 0
    cur_passes = 0
    eps = 10e-5
    time_limit = 4
    start_time = time.process_time()
    while (cur_passes < passes_limit) & (time.process_time() < time_limit + start_time):
        num_changes = 0
        for i in range(num_objects):
            x_i, y_i = parse_object(i, dataset)
            E_i = calc_f(x_i, dataset, alphas, beta, kernel_function) - y_i
            if ((y_i * E_i < -tolerance) & (alphas[i] < C)) | ((y_i * E_i > tolerance) & (alphas[i] > 0)):
                j = random_except(num_objects, i)
                x_j, y_j = parse_object(j, dataset)
                E_j = calc_f(x_j, dataset, alphas, beta, kernel_function) - y_j
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
                l, h = calc_limits(y_i, y_j, alphas[i], alphas[j], C)
                if l == h:
                    continue
                eta = 2 * kernel_function(x_i, x_j) - kernel_function(x_i, x_j) - kernel_function(x_j, x_j)
                if eta >= 0:
                    continue
                alphas[j] -= y_j * (E_i - E_j) / eta
                alphas[j] = np.clip(alphas[j], l, h)
                if abs(alphas[j] - alpha_j_old) < eps:
                    continue
                alphas[i] += y_i * y_j * (alpha_j_old - alphas[j])
                beta_1 = beta - E_i - y_i * (alphas[i] - alpha_i_old) * kernel_function(x_i, x_i) - y_j * (
                        alphas[j] - alpha_j_old) * kernel_function(x_i, x_j)
                beta_2 = beta - E_j - y_i * (alphas[i] - alpha_i_old) * kernel_function(x_i, x_j) - y_j * (
                        alphas[j] - alpha_j_old) * kernel_function(x_j, x_j)
                beta = update_beta(beta_1, beta_2, alphas[i], alphas[j], C)
                num_changes += 1
        cur_passes += num_changes == 0

    return alphas, beta


def calc_accuracy(train_dataset, test_dataset, alphas, beta, kernel_function):
    true = 0
    for i in range(len(test_dataset)):
        x_i, target = parse_object(i, test_dataset)
        predicted = np.sign(calc_f(x_i, train_dataset, alphas, beta, kernel_function))
        true += (predicted == target)
    return true / len(test_dataset)


def test_linear_kernel(dataset, Cs):
    kf = KFold(n_splits=3, shuffle=True)
    results = []
    for C in Cs:
        accuracies = []
        for train_index, test_index in kf.split(dataset):
            train_dataset = dataset[train_index]
            test_dataset = dataset[test_index]
            K = linear_kernel
            alphas, beta = sequential_minimal_optimization(train_dataset, K, C)
            accuracy = calc_accuracy(train_dataset, test_dataset, alphas, beta, K)
            accuracies.append(accuracy)
        avg = np.mean(accuracies)
        results.append([avg, C])
        print('linear kernel, C = {}, accuracy = {}'.format(C, avg))
    return results


def test_parametrized_kernel(dataset, Cs, kernel_function, kernel_params, kernel_param_name=None):
    kf = KFold(n_splits=3, shuffle=True)
    results = []
    for kernel_param in kernel_params:
        for C in Cs:
            accuracies = []
            for train_index, test_index in kf.split(dataset):
                train_dataset = dataset[train_index]
                test_dataset = dataset[test_index]
                K = kernel_function(kernel_param)
                alphas, beta = sequential_minimal_optimization(train_dataset, K, C)
                accuracy = calc_accuracy(train_dataset, test_dataset, alphas, beta, K)
                accuracies.append(accuracy)
            avg = np.mean(accuracies)
            results.append([avg, C, kernel_param])
            print('{}, C = {}, {} = {}, accuracy = {}'.format(
                kernel_function.__name__, C, kernel_param_name, kernel_param, avg)
            )
    return results


def draw_plot(title, dataset, alphas, beta, kernel_function):
    plt.title(title)
    min_x1, max_x1 = np.min(dataset[:, 0]), np.max(dataset[:, 0])
    min_x2, max_x2 = np.min(dataset[:, 1]), np.max(dataset[:, 1])
    dx1 = max_x1 - min_x1
    dx2 = max_x2 - min_x2
    step1 = 10
    step2 = 30
    for x1 in np.arange(min_x1 - dx1 / step1, max_x1 + dx1 / step1, dx1 / step2):
        for x2 in np.arange(min_x2 - dx2 / step1, max_x2 + step1 / step1, dx2 / step2):
            z = beta + np.sum(
                [alphas[i] * dataset[i][-1] * kernel_function(dataset[i][:-1], [x1, x2]) for i in range(len(dataset))]
            )
            c = 'w'
            c = 'lightcoral' if z > 0 else c
            c = 'lavenderblush' if z < 0 else c
            plt.scatter(x1, x2, color=c, s=150, marker='s')
    for i in range(len(dataset)):
        features, label = parse_object(i, dataset)
        c = 'orchid' if label == -1 else 'red'
        plt.scatter(features[0], features[1], color=c, s=50)
    plt.show()


def process_results(results):
    results.sort(reverse=True)
    num_best = 5
    best_params = [results[i] for i in range(num_best)]
    return best_params


def process_dataset(dataset):
    Cs = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    poly_degrees = [2, 3, 4, 5]
    betas = [1, 2, 3, 4, 5]

    gaussian_results = test_parametrized_kernel(dataset, Cs, gaussian_kernel, betas, 'beta')
    polynomial_results = test_parametrized_kernel(dataset, Cs, polynomial_kernel, poly_degrees, 'degree')
    linear_results = test_linear_kernel(dataset, Cs)

    best_linear = process_results(linear_results)
    best_gaussian = process_results(gaussian_results)
    best_polynomial = process_results(polynomial_results)

    print('linear: ', best_linear)
    print('gaussian: ', best_gaussian)
    print('polynomial: ', best_polynomial)

    alphas, beta = sequential_minimal_optimization(dataset, linear_kernel, C=50)
    draw_plot('linear, C = 50', dataset, alphas, beta, linear_kernel)

    alphas, beta = sequential_minimal_optimization(dataset, gaussian_kernel(1), C=1)
    draw_plot('gaussian, beta = 1, C = 1', dataset, alphas, beta, gaussian_kernel(1))

    alphas, beta = sequential_minimal_optimization(dataset, polynomial_kernel(2), C=5)
    draw_plot('polynomial, degree = 2, C = 5', dataset, alphas, beta, polynomial_kernel(2))


def read_dataset(f):
    labels = {'P': 1., 'N': -1.}
    f.readline()
    content = f.read().split()
    num_objects = len(content)
    dataset = list()
    for i in range(num_objects):
        line = content[i].split(',')
        dataset.append(list(map(float, line[:-1])) + [labels[line[-1]]])
    return np.array(dataset)


def run(filename):
    f = open(filename, 'r')
    dataset = read_dataset(f)
    process_dataset(dataset)


if __name__ == '__main__':
    run('datasets/chips.csv')
