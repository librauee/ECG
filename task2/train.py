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
import entropy as ent
from scipy.stats import skew, kurtosis
from scipy.signal import resample
import re
import os
import sklearn
import glob
# import biosppy
import ecg
import scipy.io as sio

warnings.filterwarnings('ignore')

# 读取label 以及文件路径
f = open("/datasets/heart/task2/trainreference.csv", "r")
data = [line.strip().split(',') for line in f]

dic_list = []
for d in data:
    dic = {i: 1 for i in d[1:]}
    dic['name'] = d[0]
    dic_list.append(dic)
print(dic_list[:10])
data = pd.DataFrame(dic_list)
data = data.fillna(0)
print(data[:10])

data.columns = [f'ill_{i}' if i.isdigit() else 'name' for i in data.columns]
# 1, name, 3, 11, 12, 5, 10, 2, 8, 4, 7, 9, 6
print(data[:10])
for f in data.columns:
    print(data[f].value_counts())

train_path = glob.glob('/datasets/heart/task2/Train/*.mat')
train_path = sorted(train_path)
print(train_path[:10])


# 特征提取
if not os.path.exists('feature_base.pkl'):

    tmp_array = []
    resample_num = 50

    for path in tqdm(train_path):

        ecgdata = sio.loadmat(path)['ecgdata']
        tmp_list = [os.path.basename(path)[:-4]]

        fs = 500
        for i in range(12):
            try:
                # ss = biosppy.signals.ecg.ecg(ecgdata[i], show=False, sampling_rate=fs)
                ss = ecg.ecg(ecgdata[i], show=False, sampling_rate=fs)

                # 心率变异性特征 R波 RR间期标准差、最大最小均值、采样熵、pNN50、RMSSD
                rpeaks = ss['rpeaks']
                if rpeaks.shape[0] != 0:
                    rr_intervals = np.diff(rpeaks)
                    min_dis = rr_intervals.min()
                    drr = np.diff(rr_intervals)
                    r_density = (rr_intervals.shape[0] + 1) / ecgdata.shape[1] * fs
                    pnn50 = drr[drr >= fs * 0.05].shape[0] / rr_intervals.shape[0]
                    rmssd = np.sqrt(np.mean(drr * drr))
                    samp_entrp = ent.sample_entropy(rr_intervals, 2, 0.2 * np.std(rr_intervals))
                    samp_entrp[np.isnan(samp_entrp)] = -2
                    samp_entrp[np.isinf(samp_entrp)] = -1
                    tmp_list.extend([rr_intervals.min(),
                                     rr_intervals.max(),
                                     rr_intervals.mean(),
                                     rr_intervals.std(),
                                     skew(rr_intervals),
                                     kurtosis(rr_intervals),
                                     r_density, pnn50, rmssd, samp_entrp[0], samp_entrp[1]
                                     ])


                else:
                    tmp_list.extend([np.nan] * 11)

                heart_rate = ss['heart_rate']
                if heart_rate.shape[0] != 0:
                    tmp_list.extend([heart_rate.min(),
                                     heart_rate.max(),
                                     heart_rate.mean(),
                                     heart_rate.std(),
                                     skew(heart_rate),
                                     kurtosis(heart_rate)])
                else:
                    tmp_list.extend([np.nan] * 6)

                # QRS波形态特征
                templates = ss['templates']
                templates_min = templates.min(axis=0)
                templates_max = templates.max(axis=0)
                templates_diff = templates_max - templates_min
                templates_mean = templates.mean(axis=0)
                templates_std = templates.std(axis=0)
                for j in [templates_diff, templates_mean, templates_std]:
                    tmp_rmp = resample(j, num=resample_num)
                    tmp_list.extend(list(tmp_rmp))
            except:
                tmp_list.extend([np.nan] * (resample_num * 3 + 17))

            # # HOS特征，高阶统计量
            # skew_list = []
            # kurtosis_list = []
            # sep = len(ecgdata[i]) // 5
            # for j in range(5):
            #     tmp = ecgdata[i][sep * j: sep * (j + 1)]
            #     skew_list.append(skew(tmp))
            #     kurtosis_list.append(kurtosis(tmp))
            # tmp_list.extend([np.mean(skew_list), np.mean(skew_list)])

        tmp_array.append(tmp_list)


    d = pd.DataFrame(tmp_array)
    d.rename(columns={0: 'name'}, inplace=True)
    del tmp_array
    d.to_pickle('feature_base.pkl')
else:
    d = pd.read_pickle('feature_base.pkl')

train = pd.merge(data, d, on='name', how='left')

def train_model(X_train, features, type_, is_weight=False, save_model=False):
    """
    训练lgb模型
    """
    y = X_train[type_]

    fold = 5
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=fold, random_state=2021, shuffle=True)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 6,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 5000,
        'max_bin': 255,
        'verbose': -1,
        'seed': 2021,
        'bagging_seed': 2021,
        'feature_fraction_seed': 2021,
        'early_stopping_rounds': 50,
        'is_unbalance': is_weight
    }
    oof_lgb = np.zeros(len(X_train))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
        num_round = 10000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=10,
            early_stopping_rounds=50,
            # categorical_feature=cat_cols

        )

        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        feat_imp_df['imp'] += clf.feature_importance() / fold
        if save_model:
            clf.save_model(f'model_only/{type_}_model_{fold_}.txt')

    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

    return feat_imp_df, oof_lgb

tags = [f'ill_{i}' for i in range(1, 13)]
for f in tags:

    features = [i for i in d.columns if i not in ['tag', 'name', ] + tags]
    if f[-1] in ['4', '7', '8'] or f[-2:] in ['11', '12']:
        feat_imp_df, oof_lgb = train_model(train, features, f, True, True)
    else:
        feat_imp_df, oof_lgb = train_model(train, features, f, False, True)