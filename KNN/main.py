import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from utils import manhattan, euclidean, chebyshev
from utils import uniform, triangular, epanechnikov, quartic, triweight, tricube, gaussian, cosine, logistic, sigmoid


def minmax(dataset):
    num_features = len(dataset[0]) - 1
    return [[dataset[:, i].min(), dataset[:, i].max()] for i in range(num_features)]


def normalize(dataset, min_max):
    num_features = len(dataset[0]) - 1
    for obj in dataset:
        for i in range(num_features):
            min_value = min_max[i][0]
            max_value = min_max[i][1]
            obj[i] = (obj[i] - min_value) / (max_value - min_value)
    return dataset


def enumerate_labels(dataset):
    label_by_ind = list(sorted(set(dataset[:, -1])))
    ind_by_label = dict(zip(label_by_ind, [i for i in range(len(label_by_ind))]))
    for i in range(len(dataset)):
        dataset[i][-1] = ind_by_label[dataset[i][-1]]
    return [label_by_ind, ind_by_label]


def get_F_score(cm):
    num_objects = np.sum(cm)
    F_value = 0
    recall, precision = 0, 0
    safe_div = lambda x, y: 0 if y == 0 else x / y

    for c in range(len(cm)):
        numer = np.sum(np.array(cm)[c])
        denomer = np.sum(np.array(cm)[:, c])

        recall_c = safe_div(cm[c][c], numer)
        precision_c = safe_div(cm[c][c], denomer)

        weighted_recall_c = numer * recall_c
        weighted_precision_c = numer * precision_c

        recall += weighted_recall_c
        precision += weighted_precision_c

        F_value += safe_div(numer * 2 * recall_c * precision_c, recall_c + precision_c)

    return F_value / num_objects


def predict(dataset, target, distance_function, kernel_function, window_type, window_param):
    num_objects = len(dataset)

    if window_type == 'variable':
        distances = [distance_function(dataset[i], target) for i in range(num_objects)]
        distances.sort()
        window_param = distances[window_param]

    numer, denomer = 0, 0

    for i in range(num_objects):
        distance = distance_function(dataset[i], target)
        w = 0
        if window_param != 0:
            w = kernel_function(distance / window_param)
        elif (window_param == 0) & (distance == 0):
            w = kernel_function(0)
        numer += dataset[i][-1] * w
        denomer += w

    prediction = numer / denomer if denomer != 0 else np.sum(dataset, axis=0)[-1] / num_objects
    return prediction


def knn(dataset, target, distance_function, kernel_function, window_type, window_param):
    min_max = minmax(dataset)

    dataset = normalize(dataset, min_max)
    target = normalize([target], min_max)[0]

    return predict(dataset, target, distance_function, kernel_function, window_type, window_param)


def cross_validation(dataset_original, distance_function, kernel_function, window_type, window_param, get_prediction):
    dataset_original = np.array(dataset_original)
    num_objects = len(dataset_original)
    num_classes = len(set(dataset_original[:, -1]))

    enumerate_labels(dataset_original)
    cm = [[0 for i in range(num_classes)] for j in range(num_classes)]

    for i in range(num_objects):
        dataset = np.delete(dataset_original, i, 0)
        target = dataset_original[i].copy()
        target_label = target[-1]
        target[-1] = -1
        predicted_label = get_prediction(num_classes, dataset, target, distance_function,
                                         kernel_function, window_type, window_param)
        cm[int(target_label)][int(predicted_label)] += 1

    return get_F_score(cm)


def naive_calc_predicted(_, dataset, target, distance_function, kernel_function, window_type, window_param):
    return int(knn(dataset, target, distance_function, kernel_function, window_type, window_param))


def naive_cross_validation(dataset_original, distance_function, kernel_function, window_type, window_param):
    return cross_validation(dataset_original, distance_function, kernel_function,
                            window_type, window_param, naive_calc_predicted)


def one_hot_calc_predicted(num_classes, dataset, target, distance_function, kernel_function, window_type, window_param):
    resulting_vector = list()
    for clazz in range(num_classes):
        class_dataset = dataset.copy()
        for obj in class_dataset:
            obj[-1] = 1 if obj[-1] == clazz else 0
        prediction = knn(class_dataset, target.copy(), distance_function, kernel_function, window_type, window_param)
        resulting_vector.append(prediction)
    return np.argmax(resulting_vector)


def one_hot_cross_validation(dataset_original, distance_function, kernel_function, window_type, window_param):
    return cross_validation(dataset_original, distance_function, kernel_function,
                            window_type, window_param, one_hot_calc_predicted)


def train(dataset, validation):
    distance_functions = [manhattan, euclidean, chebyshev]
    kernel_functions = [uniform, triangular, epanechnikov, quartic, triweight,
                        tricube, gaussian, cosine, logistic, sigmoid]
    window_types = ['fixed', 'variable']
    params = {
        'fixed': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5],
        'variable': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 27, 35, 50]
    }

    all_F = list()
    max_F, max_args = 0, list()
    eps = 1e-6

    for window_type in window_types:
        for distance_function in distance_functions:
            for kernel_function in kernel_functions:
                F_values = list()
                for window_param in params[window_type]:
                    distance_func_name = distance_function.__name__
                    kernel_func_name = kernel_function.__name__

                    F = validation(dataset, distance_function, kernel_function, window_type,
                                   window_param)
                    F_values.append(F)

                    if (F > max_F) | (abs(max_F - F) < eps):
                        if F > max_F:
                            max_F = F
                            max_args.clear()
                        max_args.append((distance_func_name, kernel_func_name, window_type, window_param, F))

                    print('distance = {}, kernel = {}, window_type = {}, window_param = {}, F = {}'
                          .format(distance_func_name, kernel_func_name, window_type, window_param, F))

                all_F.append(F_values)
        draw_plot(params[window_type], all_F)

    return max_args


def naive(dataset):
    return train(dataset, naive_cross_validation)


def one_hot(dataset):
    return train(dataset, one_hot_cross_validation)


def draw_plot(params, lines):
    x = params
    for i in range(len(lines)):
        y = lines[i]
        plt.plot(x, y)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.show()


def draw_resulting_lines(x, y1, y2):
    plt.plot(x, y1, label='{}: {}, {}'.format('one-hot', 'manhattan', 'triweight'))
    plt.plot(x, y2, label='{}: {}, {}'.format('naive', 'manhattan', 'triangular'))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fancybox=True, shadow=True)
    plt.xlabel('h')
    plt.ylabel('F')
    plt.show()


def draw_results_variable():
    x = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 27, 35, 50]

    # selected one-hot: manhattan, triweight (the best: 9 0.758956256619808)
    y1 = [0.7346140213300709, 0.7392807861588931, 0.7434292814377058, 0.7509405123062826,
          0.7541769734628638, 0.7586509880077774, 0.7537193018350691, 0.758956256619808, 0.7493636420361415,
          0.7391572834370127, 0.7393683994907576, 0.7346240203477407, 0.7282952960596141, 0.7096412358680698,
          0.7181118441192381, 0.6871761962407122, 0.6794074231673245]

    # selected naive: manhattan, triangular (the best: 2 0.6305270645317834)
    y2 = [0.6305270645317834, 0.5842273706367677, 0.48644425797864643, 0.4636237304324165,
          0.44529272711785894, 0.4059408750619823, 0.39896905877096434, 0.3455506199225732, 0.3185349763520958,
          0.30519672223076283, 0.26134385477995814, 0.2602287195811969, 0.19679230426893976, 0.1647525407026965,
          0.12815627752787317, 0.13270969644416028, 0.13717331201959368]

    draw_resulting_lines(x, y1, y2)


def draw_results_fixed():
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5]

    # selected one-hot: manhattan, gaussian (the best 0.1 0.7382181445399424)
    y1 = [0.7382181445399424, 0.6952615426515586, 0.6997030335493596, 0.6577184256285126, 0.639258168753818,
          0.6026668129471868, 0.5615399203819328, 0.523541598765011, 0.4941633609666732, 0.4478654842691453,
          0.23915378961057845, 0.20943541680116357, 0.1849688473520249]

    # selected naive: manhattan, triweight (the best 0.1 0.37673030954657316)
    y2 = [0.37673030954657316, 0.31032966599629325, 0.34504644389611355, 0.3340491308312156, 0.34212505371723256,
          0.3724660713565786, 0.33866117300427656, 0.3221697766869106, 0.27625097518677566, 0.26051554329780796,
          0.20502938512834057, 0.20249754705200249, 0.15945558922533812]

    draw_resulting_lines(x, y1, y2)


def main():
    filename = 'datasets/prnn_fglass.csv'
    dataset = pd.read_csv(filename).values
    naive(dataset)
    one_hot(dataset)
    draw_results_variable()
    draw_results_fixed()


if __name__ == '__main__':
    main()
