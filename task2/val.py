import pandas as pd
import json
from tqdm import tqdm
import warnings
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
# import pyentrp.entropy as ent
from scipy.stats import skew, kurtosis
from scipy.signal import resample
import re
import os
import sklearn
import glob
# import biosppy
import scipy.io as sio
import sys


warnings.filterwarnings('ignore')


# 训练集label
f = open("/datasets/heart/task2/trainreference.csv", "r")
data = [line.strip().split(',') for line in f]
dic_list = []
for d in data:
    dic = {i: 1 for i in d[1:]}
    dic['name'] = d[0]
    dic_list.append(dic)
data = pd.DataFrame(dic_list)
data = data.fillna(0)
data.columns = [f'ill_{i}' if i.isdigit() else 'name' for i in data.columns]

# 特征
d = pd.read_pickle('feature_base.pkl')
train = pd.merge(data, d, on='name', how='left')
del d

tags = [f'ill_{i}' for i in range(1, 13)]
features = [i for i in train.columns if i not in ['tag', 'name'] + tags]


# 生成预测oof
def infer(data, f, return_prob=True):
    y = train[f]
    oof_lgb = np.zeros(len(data))

    KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
    for fold_, (trn_idx, val_idx) in enumerate(KF.split(data.values, y.values)):
        clf = lgb.Booster(model_file=f'model/{f}_model_{fold_}.txt')
        oof_lgb[val_idx] = clf.predict(data.iloc[val_idx][features], num_iteration=clf.best_iteration)
    if return_prob:
        data[f] = oof_lgb
    else:
        data[f] = [1 if i > 0.5 else 0 for i in oof_lgb]
    return data


for f in tqdm(tags):
    d = infer(d, f)

d[['name'] + tags].to_csv('oof.csv', index=False)
data[['name'] + tags].to_csv('label.csv', index=False)


# 线下得分计算
# f1_score
f1_score_list = []
for f in tqdm(tags):
    d = infer(d, f, False)
    f1_score_list.append(f1_score(train[f], d[f]))

print(f1_score_list)
print(np.mean(f1_score_list))

# 完全匹配得分
s = data[tags].values == d[tags].values
s = np.sum(s, axis=1)
s = sum(np.where(s == 12, 1, 0))

print(s / len(data))


# 优化- 疾病标签全为0则正常标签置为1，疾病标签非全0则正常标签置为0
f1_score_list2 = []
for i in range(1, 13):
    d['sum_count'] = d[[f'ill_{i}' for i in range(2, 13)]].sum(1)
    d['ill_1'] = d['sum_count'].apply(lambda x: 1 if x == 0 else 0)
    f1_score_list2.append(f1_score(train[f'ill_{i}'], d[f'ill_{i}']))

print(f1_score_list2)
print(np.mean(f1_score_list2))

s = data[tags].values == d[tags].values
s = np.sum(s, axis=1)
s = sum(np.where(s == 12, 1, 0))

print(s / len(data))