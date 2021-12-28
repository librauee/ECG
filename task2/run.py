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
import sys

warnings.filterwarnings('ignore')


# 提取测试集数据特征
if not os.path.exists('test_feature_base_.pkl'):
    path = sys.argv[1]
    test_path = glob.glob(os.path.join(path, '*.mat'))
    test_path = sorted(test_path)
    tmp_array = []
    resample_num = 50

    for path in tqdm(test_path):

        ecgdata = sio.loadmat(path)['ecgdata']
        tmp_list = [os.path.basename(path)[:-4]]

        fs = 500
        for i in range(12):
            try:
                ss = ecg.ecg(ecgdata[i], show=False, sampling_rate=fs)
                rpeaks = ss['rpeaks']
                if rpeaks.shape[0] != 0:
                    rr_intervals = np.diff(rpeaks)
                    min_dis = rr_intervals.min()
                    drr = np.diff(rr_intervals)
                    r_density = (rr_intervals.shape[0] + 1) / ecgdata.shape[0] * fs
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


    test = pd.DataFrame(tmp_array)
    test.rename(columns={0: 'name'}, inplace=True)
    del tmp_array
    test.to_pickle('test_feature_base.pkl')
else:
    test = pd.read_pickle('test_feature_base.pkl')


# 推理以及加阈值后处理
def infer(data, f):
    predictions_lgb = np.zeros((len(data)))
    weight_dic = {
        'ill_1': 0.5,
        'ill_2': 0.4115,
        'ill_3': 0.5,
        'ill_4': 0.492,
        'ill_5': 0.5,
        'ill_6': 0.5,
        'ill_7': 0.5,
        'ill_8': 0.5,
        'ill_9': 0.5,
        'ill_10': 0.5,
        'ill_11': 0.5,
        'ill_12': 0.406,
    }
    ff = f
    for i in range(5):
        clf = lgb.Booster(model_file=f'model_with_nn/{ff}_model_{i}.txt')
        y_pred = clf.predict(data[features], num_iteration=clf.best_iteration)
        predictions_lgb[:] += y_pred / 5
    # data[f] = [1 if i > 0.5 else 0 for i in predictions_lgb]
    data[f] = [1 if i > weight_dic[ff] else 0 for i in predictions_lgb]
    return data

nn_pred = pd.read_csv('test_pred.csv')
nn_pred.columns = ['name'] + [f'ill_{i}_nn' for i in range(1, 13)]
test = pd.merge(test, nn_pred, on='name', how='left')

tags = [f'ill_{i}' for i in range(1, 13)]
features = [i for i in test.columns if i not in ['tag', 'name', ] + tags]
for f in tags[1:]:
    test = infer(test, f)

test['sum_count'] = test[[f'ill_{i}' for i in range(2, 13)]].sum(1)
test['ill_1'] = test['sum_count'].apply(lambda x: 1 if x == 0 else 0)
test[['name'] + tags].to_csv('result_base.csv', index=False)

for f in tags:
    print(test[f].value_counts())

for i in range(1, 13):
    test[f'ill_{i}'] = test[f'ill_{i}'].apply(lambda x: str(i) if x == 1 else 'mask')

# 写入提交文件
with open('answer.csv', 'w', encoding='utf-8') as f:
    for i in range(len(test)):
        s = ','.join(test[['name'] + [f'ill_{j}' for j in range(1, 13)]].values[i])
        f.write(s.replace('mask,', '').strip('mask,') + '\n')