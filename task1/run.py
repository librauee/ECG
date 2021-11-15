import pandas as pd
import json
from tqdm import tqdm
import warnings
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import re
import os
import sklearn
import glob
import scipy.io as sio

warnings.filterwarnings('ignore')

sep_length = 10

def get_data(sep_length):
    """
    读取数据，提取部分数据用于训练预测
    """

    train_path = glob.glob('data/train/*.mat')
    test_path = glob.glob('data/val/*.mat')

    matrix = np.zeros((2000, 5000 * 12 // sep_length))

    i = 0
    for p in tqdm(train_path):
        ecgdata = sio.loadmat(p)['ecgdata']
        sep = ecgdata[:, ::sep_length]
        sep = sep.reshape(-1,)
        matrix[i, :] = sep
        i += 1
    for p in tqdm(test_path):
        ecgdata = sio.loadmat(p)['ecgdata']
        sep = ecgdata[:, ::sep_length]
        sep = sep.reshape(-1,)
        matrix[i, :] = sep
        i += 1  

    data = pd.DataFrame(matrix)
    data.columns = [f'{i}' for i in range(len(data.columns))]
    return data

def feature_eng(data, f, feat):
    """
    统计特征
    """
    data[f'{f}_min'] = data[feat].min(1)
    data[f'{f}_max'] = data[feat].max(1)
    data[f'{f}_meam'] = data[feat].mean(1)
    data[f'{f}_std'] = data[feat].std(1)
    data[f'{f}_median'] = data[feat].median(1)
    data[f'{f}_skew'] = data[feat].skew(1)    
    return data
  
def train_model(X_train, X_test, features, y, save_model=False):
    """
    训练lgb模型
    """
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
    params = {
        'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.05,
                    'num_leaves': 2**6,
                    'max_depth':8,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.8,
                    'subsample_freq':1,
                    'subsample':0.8,
                    'num_boost_round':5000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': 2021,
                    'bagging_seed': 2021,
                    'feature_fraction_seed': 2021,
                    'early_stopping_rounds':100,
    }
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros((len(X_test)))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
        num_round = 10000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=50,
            # categorical_feature=cat_cols

        )

        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration) / 5
        feat_imp_df['imp'] += clf.feature_importance() / 5
        if save_model:
            clf.save_model(f'model_{fold_}.txt')
        

    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

    return feat_imp_df, oof_lgb, predictions_lgb

data = get_data(sep_length)
feat_ori = data.columns
feat_len = len(data.columns)
data = feature_eng(data, f='origin', feat=[f'{i}' for i in range(feat_len)])
data[[f'{i}' for i in range(feat_len, feat_len * 2)]] = data[feat_ori].diff(1, axis=1)
data = feature_eng(data, f='diff', feat=[f'{i}' for i in range(feat_len, feat_len * 2)])
data = feature_eng(data, f='total', feat=[f'{i}' for i in range(feat_len * 2)])


for i in tqdm(range(12 * 2)):
    sep_num = 5000 // sep_length
    feat = [str(j) for j in range(sep_num * i, sep_num * (i + 1))]
    data = feature_eng(data, f=str(i), feat=feat)

    
train_path = [i.split('\\')[-1].split('.')[0] for i in train_path]
test_path = [i.split('\\')[-1].split('.')[0] for i in test_path]
data['name'] = train_path + test_path
labels = pd.read_csv('data/trainreference.csv')
data = pd.merge(data, labels, on='name', how='left')

train = data[~data['tag'].isna()].reset_index(drop=True)
test = data[data['tag'].isna()].reset_index(drop=True)

features = [i for i in train.columns if i not in ['tag', 'name',]]
y = train['tag']

feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y)

test['tag'] = [1 if i > 0.5 else 0 for i in predictions_lgb]
test[['name', 'tag']].to_csv('answer.csv', index=False)
