import random
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, rand_score


def minmax(xs):
    num_features = len(xs[0])
    return [[xs[:, i].min(), xs[:, i].max()] for i in range(num_features)]


def normalize(xs, min_max):
    num_features = len(xs[0])
    for obj in xs:
        for i in range(num_features):
            min_value = min_max[i][0]
            max_value = min_max[i][1]
            obj[i] = (obj[i] - min_value) / (max_value - min_value)
    return xs


def enumerate_labels(labels):
    label_by_ind = list(sorted(set(labels)))
    ind_by_label = dict(zip(label_by_ind, [i for i in range(len(label_by_ind))]))
    for i in range(len(labels)):
        labels[i] = ind_by_label[labels[i]]
    return [label_by_ind, ind_by_label]


def k_means(xs: np.ndarray, k: int):
    objects_num, features_num = xs.shape

    centers = random.sample(list(xs), k)
    clusters = [-1 for _ in range(objects_num)]

    changed = True
    while changed:
        changed = False
        for i in range(objects_num):
            norms = [np.linalg.norm(xs[i] - centers[j]) for j in range(k)]
            new_cluster = int(np.argmin(norms))
            if new_cluster != clusters[i]:
                changed = True
                clusters[i] = new_cluster
        for j in range(k):
            numerator = np.array([np.array(xs[i]) * (clusters[i] == j) for i in range(objects_num)]).sum(axis=0)
            denominator = np.array([(clusters[i] == j) for i in range(objects_num)]).sum(axis=0)
            centers[j] = numerator / denominator

    return clusters


def calc_rand_index(ys_predicted, ys_target):
    objects_num = len(ys_predicted)
    TP, TN = 0, 0
    FP, FN = 0, 0
    for i in range(objects_num):
        for j in range(i + 1, objects_num):
            same_cluster = (ys_predicted[i] == ys_predicted[j])
            same_class = (ys_target[i] == ys_target[j])
            TP += (same_cluster & same_class)
            TN += same_cluster & (not same_class)
            FP += (not same_cluster) & same_class
            FN += (not same_cluster) & (not same_class)
    return (TP + FN) / (TP + TN + FP + FN)


def calc_a(i, c_k, xs, ys, norm):
    objects_num = len(xs)
    s = 0
    counter = 0
    for j in range(objects_num):
        if ys[j] != c_k:
            continue
        counter += 1
        s += norm[i][j]
    return s / counter


def calc_b(i, c_k, xs, ys, norm):
    clusters_num = len(set(ys))
    result = -1
    for c_l in range(clusters_num):
        if c_k == c_l:
            continue
        a = calc_a(i, c_l, xs, ys, norm)
        if result == -1:
            result = a
        result = min(result, a)
    return result


def calc_silhouette(xs, ys):
    objects_num = len(xs)
    clusters_num = len(set(ys))
    norm = [[np.linalg.norm(xs[i] - xs[j]) for j in range(objects_num)] for i in range(objects_num)]
    s = 0
    for c_k in range(clusters_num):
        for i in range(objects_num):
            if ys[i] != c_k:
                continue
            b = calc_b(i, c_k, xs, ys, norm)
            a = calc_a(i, c_k, xs, ys, norm)
            s += (b - a) / max(a, b)
    return s / objects_num


def draw_clusters(xs, ys, title):
    labels_num = len(set(ys))
    groups = [[] for _ in range(labels_num)]
    colors = ['deeppink', 'orange', 'cyan']
    # colors = ['red', 'green', 'blue']
    for i in range(len(xs)):
        groups[ys[i]].append(xs[i])
    for i in range(labels_num):
        x = np.array(groups[i])[:, 0]
        y = np.array(groups[i])[:, 1]
        plt.plot(x, y, 'ro', c=colors[i], markersize=3)
    plt.title(title)
    plt.show()


def draw_plot(k_values, rand_scores, silhouette_scores):
    plt.plot(k_values, rand_scores, label='rand_score')
    plt.plot(k_values, silhouette_scores, label='silhouette_score')
    plt.legend()
    plt.xlabel('k_value')
    plt.ylabel('metrics_value')
    plt.show()


def main():
    filename = 'datasets/dataset_191_wine.csv'
    dataset = pd.read_csv(filename).values

    xs = np.array(dataset[:, 1:])
    ys_target = np.array(list(map(int, dataset[:, 0])))
    enumerate_labels(ys_target)

    min_max = minmax(xs)
    xs = normalize(xs, min_max)

    results = {'k': [], 'rand_score': [], 'silhouette_score': []}
    k_values = [i for i in range(2, 15)]

    for k in k_values:
        ys_predicted = k_means(xs, k)
        rand = rand_score(ys_target, ys_predicted)
        silhouette = silhouette_score(xs, ys_predicted)
        print('k = {}, rand_score = {}, silhouette_score = {}'.format(k, rand, silhouette))
        results['k'].append(k)
        results['rand_score'].append(rand)
        results['silhouette_score'].append(silhouette)

    draw_plot(results['k'], results['rand_score'], results['silhouette_score'])
    table = pd.DataFrame.from_dict(results)
    table.to_csv('results/{}.csv'.format('wine_results'))

    pca = PCA(n_components=2)
    reduced_xs = pca.fit_transform(xs)
    draw_clusters(reduced_xs, ys_target, 'Target decomposition, wine')
    draw_clusters(reduced_xs, k_means(xs, 3), 'Predicted decomposition, wine')


if __name__ == '__main__':
    main()
