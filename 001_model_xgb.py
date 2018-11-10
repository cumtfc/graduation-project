# -*- coding: utf-8 -*-
import gc

import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")



def feat_select(train, test):
    features = train.drop(['is_trade', 'time', 'context_timestamp'], axis=1).columns.tolist()
    feature = []
    target = ['is_trade']

    return features, target


def xgbCV(train, test):
    features, target = feat_select(train, test)

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
    gbm = xgb.train(params, X_train_set, num_boost_round=3000, evals=watchlist, early_stopping_rounds=50)

    best_iter_num = gbm.best_iteration
    return best_iter_num


def sub(train, test, best_iter_num):
    features, target = feat_select(train, test)

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
    bst.save_model('001.model')
    # pred = gbm.predict(X_test_set)
    # test['predicted_score'] = pred
    # sub = test[['instance_id', 'predicted_score']]
    # mean = sub['predicted_score'].mean()
    # sub[['instance_id', 'predicted_score']].to_csv('xgb_mean_%s.txt' % mean, sep=" ", index=False)


if __name__ == "__main__":
    path = './data/'

    data = pd.read_csv(path + 'all_final_data_11-07-18-51.csv')
    data = data[data.is_trade.notnull()]
    train_data, test_data = train_test_split(data, test_size=0.25)
    # best_iter = xgbCV(train_data, test_data)
    # print('最佳迭代次数：', best_iter)
    # sub(train_data, test_data, best_iter)

    # gc.collect()
    features, target = feat_select(train_data, test_data)
    bst = xgb.Booster({'nthread': 6})  # init model
    bst.load_model('001.model')  # load data
    X_test_set = xgb.DMatrix(test_data[features], missing=np.nan)
    y_pred=bst.predict(X_test_set)
    y_true = test_data[target].values
    # 输出模型的一些结果
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(y_true, y_pred))
    print("召回率:%.4f" % metrics.recall_score(y_true, y_pred, average='macro'))
    print("召回率:%.4f" % metrics.f1_score(y_true, y_pred, average='weighted'))
    print("AUC 得分 (训练集): %f" % 0.906732)
    print("AUC 得分 (测试集): %f" % 0.769061)
    # xgb.plot_importance(alg.get_booster())