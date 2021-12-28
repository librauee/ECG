import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

data = pd.read_csv('label.csv')
d = pd.read_csv('oof.csv')

tags = [f'ill_{i}' for i in range(1, 13)]

def get_score_other(f, threshold_list):
    score = 0
    for i in range(len(tags)):
        if tags[i] != f:
            score += f1_score(data[tags[i]], [1 if j > threshold_list[i] else 0 for j in d[tags[i]]])
    return score


def get_score(f, threshold, base_score, threshold_list):
    score = 0.8 * (f1_score(data[f], [1 if j > threshold else 0 for j in d[f]]) + base_score) / 12
    dd = d.copy()
    for i in range(len(tags)):
        dd[tags[i]] = dd[tags[i]].apply(lambda x: 1 if x > threshold_list[i] else 0)
    s = data[tags].values == dd[tags].values
    s = np.sum(s, axis=1)
    s = sum(np.where(s == 12, 1, 0)) / len(dd)
    score += s * 0.25
    return score


threshold_list = [0.5] * 12
best_threshold_list = threshold_list.copy()
best_score = 0

for i in range(len(tags) - 1, -1, -1):
    if i != 0:
        base_score = get_score_other(tags[i], threshold_list)
        threshold_list = best_threshold_list.copy()
        for j in tqdm(range(500, 1500)):
            threshold_list[i] = j / 2000
            score = get_score(tags[i], j / 2000, base_score, threshold_list)
            if score > best_score:
                best_threshold_list = threshold_list.copy()
                best_score = score

    else:
        for j in range(1, len(tags)):
            d[tags[j]] = d[tags[j]].apply(lambda x: 1 if x > threshold_list[j] else 0)

        d['sum_count'] = d[[f'ill_{i}' for i in range(2, 13)]].sum(1)
        d['ill_1'] = d['sum_count'].apply(lambda x: 1 if x == 0 else 0)
        base_score = get_score_other(tags[i], threshold_list)
        threshold_list = best_threshold_list.copy()
        score = get_score(tags[i], 0.5, base_score, threshold_list)
        if score > best_score:
            best_score = score
    print(best_threshold_list)
    print(best_score)