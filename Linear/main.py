import numpy as np
import random

from matplotlib import pyplot as plt


def minmax(dataset):
    num_features = len(dataset[0])
    return [[dataset[:, i].min(), dataset[:, i].max()] for i in range(num_features)]


def normalize(dataset, min_max):
    num_features = len(dataset[0])
    for obj in dataset:
        for i in range(num_features):
            min_value = min_max[i][0]
            max_value = min_max[i][1]
            if min_value == max_value:
                obj[i] = 0
            else:
                obj[i] = (obj[i] - min_value) / (max_value - min_value)
    return dataset


def smape(x, y):
    return np.sum([abs(a - b) / (abs(a) + abs(b)) for a, b in zip(x, y)]) / len(x)


def diff_square(obj: np.ndarray, i, w: np.ndarray):
    return 2 * obj[i] * np.dot(obj, np.append(w, -1))


def calc_gradient(obj, w):
    num_features = len(obj) - 1
    return np.array([diff_square(obj, i, w) for i in range(num_features)])


def gradient_descent_method(dataset: np.ndarray, tau=20):
    num_objects, cols = dataset.shape
    num_features = cols - 1

    num_iterations = 2000
    limit = 1 / (2 * num_features)
    w = np.array([random.uniform(-limit, limit) for _ in range(num_features)])
    w_list = list()
    smape_vals = list()
    smape_val = 1.
    l = 0.05

    for k in range(1, num_iterations + 1):
        w_prev = w
        random_object = dataset[random.randint(0, num_objects - 1)]
        grad = calc_gradient(random_object, w_prev)

        regularized_grad = grad + tau * w_prev
        eta = 1 / k
        w = w_prev - eta * regularized_grad

        target = random_object[-1]
        predicted = np.dot(random_object[:-1], w)
        smape_val = l * smape([target], [predicted]) + (1 - l) * smape_val

        smape_vals.append(smape_val)
        w_list.append(w)

    return w_list, smape_vals


def least_squares_method(dataset: np.ndarray, tau=1e-4):
    f = np.delete(dataset, -1, axis=1)
    _, num_features = f.shape
    identity = np.identity(num_features)
    f_transposed = np.transpose(f)
    y = dataset[:, -1]
    return np.linalg.inv(f_transposed @ f + tau * identity) @ f_transposed @ y


def calc_dataset_smape(dataset: np.ndarray, w):
    target = dataset[:, -1]
    predicted = np.array([np.dot(x, y) for x, y in zip(
        np.delete(dataset, -1, axis=1),
        [w for _ in range(len(dataset))]
    )])
    return smape(predicted, target)


def calc_moving_avg_smape(dataset: np.ndarray, w_list):
    num_objects, cols = dataset.shape

    smape_vals = list()
    smape_val = 1.
    l = 0.05

    for w in w_list:
        random_object = dataset[random.randint(0, num_objects - 1)]
        target = random_object[-1]
        predicted = np.dot(random_object[:-1], w)
        smape_val = l * smape([target], [predicted]) + (1 - l) * smape_val
        smape_vals.append(smape_val)

    return smape_vals


def read_dataset(f, num_objects) -> np.ndarray:
    dataset = [[1.] + list(map(float, f.readline().split())) for _ in range(num_objects)]
    min_max = minmax(np.array(dataset))
    dataset = normalize(dataset, min_max)
    return np.array(dataset)


def draw_plot(train_smape, test_smape):
    num_iterations = len(train_smape)
    iterations = [i for i in range(1, num_iterations + 1)]
    plt.plot(iterations, train_smape, label='train dataset')
    plt.plot(iterations, test_smape, label='test dataset')
    plt.legend(loc='upper center')
    plt.xlabel('iterations')
    plt.ylabel('SMAPE')
    plt.show()


def run(filename):
    f = open(filename, 'r')
    f.readline()

    train_dataset = read_dataset(f, int(f.readline()))
    test_dataset = read_dataset(f, int(f.readline()))

    w_list, smape_gd_train = gradient_descent_method(train_dataset, tau=0)
    smape_gd_test = calc_moving_avg_smape(test_dataset, w_list)
    draw_plot(smape_gd_train, smape_gd_test)

    w = least_squares_method(train_dataset)
    smape_lsm_train = calc_dataset_smape(train_dataset, w)
    smape_lsm_test = calc_dataset_smape(test_dataset, w)
    print('Train dataset, least squares method: SMAPE value = {}'.format(smape_lsm_train))
    print('Test dataset, least squares method: SMAPE value = {}'.format(smape_lsm_test))

    f.close()


if __name__ == '__main__':
    run('LR/1.txt')
