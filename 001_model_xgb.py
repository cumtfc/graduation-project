# -*- coding: utf-8 -*-
import gc

import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.combine import SMOTEENN
from win32timezone import now

warnings.filterwarnings("ignore")


def feat_select(train, test):
    features = train.drop(['is_trade', 'time', 'context_timestamp'], axis=1).columns.tolist()
    feature = []
    target = ['is_trade']

    return features, target


def xgbCV(train, test, features, target):
    X = train[features]
    y = train[target]
    X_tes = test[features]
    y_tes = test[target]

    print('Training XGB model...')
    X_train_set = xgb.DMatrix(X, label=y, missing=np.nan)
    X_validate_set = xgb.DMatrix(X_tes, label=y_tes, missing=np.nan)
    watchlist = [(X_train_set, 'train'), (X_validate_set, 'eval')]
    params = {'max_depth': 7,
              'nthread': -1,
              'eta': 0.01,
              'eval_metric': 'auc',
              'objective': 'binary:logistic',
              'subsample': 0.85,
              'colsample_bytree': 0.85,
              'silent': 1,
              'seed': 0,
              'min_child_weight': 6,
              'gpu_id': 0,
              'tree_method': 'gpu_hist'
              # 'scale_pos_weight':0.5
              }
    # gbm = xgb.train(params, X_train_set, num_boost_round=3000, evals=watchlist, early_stopping_rounds=50)
    history = xgb.cv(params, X_train_set, num_boost_round=3000, nfold=5, early_stopping_rounds=50)
    return best_iter_num


def sub(train, test, features, target, best_iter_num):
    X = train[features]
    y = train[target]
    X_train_set = xgb.DMatrix(X, label=y, missing=np.nan)
    X_test_set = xgb.DMatrix(test[features], missing=np.nan)
    print('Training XGB model...')
    params = {'max_depth': 7,
              'nthread': -1,
              'eta': 0.01,
              'eval_metric': 'auc',
              'objective': 'binary:logistic',
              'subsample': 0.85,
              'colsample_bytree': 0.85,
              'silent': 1,
              'seed': 0,
              'min_child_weight': 6,
              'gpu_id': 0,
              'tree_method': 'gpu_hist'
              # 'scale_pos_weight':0.5
              }
    bst = xgb.train(params, X_train_set, num_boost_round=best_iter_num)
    bst
    bst.save_model('001.model')
    # pred = gbm.predict(X_test_set)
    # test['predicted_score'] = pred
    # sub = test[['instance_id', 'predicted_score']]
    # mean = sub['predicted_score'].mean()
    # sub[['instance_id', 'predicted_score']].to_csv('xgb_mean_%s.txt' % mean, sep=" ", index=False)


def train():
    # train_data = rebalance()
    print("starting CV:", now())
    best_iter = xgbCV(train_data, test_data, features, target)
    print('最佳迭代次数：', best_iter)
    sub(train_data, test_data, features, target, best_iter)


def rebalance():
    sm = SMOTEENN()
    train_data.replace(to_replace=np.nan, value=0, inplace=True)
    train_data.replace(to_replace=-np.inf, value=0, inplace=True)
    train_data.replace(to_replace=np.inf, value=0, inplace=True)
    print("rebalance data:", now())
    X_resampled, y_resampled = sm.fit_resample(train_data[features], train_data[target])
    X_resampled = pd.DataFrame(X_resampled, columns=features)
    y_resampled = pd.DataFrame(y_resampled, columns=target)
    X_resampled['is_trade'] = y_resampled['is_trade']
    del y_resampled
    gc.collect()
    return X_resampled


def count():
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(predictions)):
        if predictions[i] == y_true.T[0][i]:
            if predictions[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predictions[i] == 1:
                fp += 1
            else:
                fn += 1

    print(tp, tn, fp, fn)


if __name__ == "__main__":
    path = './data/'

    data = pd.read_csv(path + 'all_final_data_11-07-18-51.csv')
    data = data[data.is_trade.notnull()]
    train_data, test_data = train_test_split(data, test_size=0.25)
    features, target = feat_select(train_data, test_data)
    # train()
    gc.collect()
    bst = xgb.Booster({'nthread': 6})  # init model
    bst.load_model('001.model')  # load data
    X_test_set = xgb.DMatrix(test_data[features], missing=np.nan)
    y_pred = bst.predict(X_test_set)
    predictions = [round(x) for x in y_pred]
    y_true = test_data[target].values
    count()
    # 输出模型的一些结果
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(y_true, predictions))
    print("召回率:%.4f" % metrics.recall_score(y_true, predictions))
    print("精度:%.4f" % metrics.precision_score(y_true, predictions))
    print("F1:%.4f" % metrics.f1_score(y_true, predictions))
    print("AUC 得分 (训练集): %f" % 0.906732)
    print("AUC 得分 (测试集): %f" % 0.769061)
    # xgb.plot_importance(alg.get_booster())
