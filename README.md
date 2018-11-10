# IJCAI18_Tianchi_Rank29
1. 001_lgbEnsemble.py lgbm 10-folds融合+反smigod
2. 001_model_lgb.py lgbm单模型训练
3. 001_model_xgb.py xgboost单模型
4. 100_process.py 数据预处理
5. 101_basic_feat.py 基础特征
6. 102_trick_feat.py 时间间隔相关特征
7. 103_statistics_feat.py 统计特征
8. 201_meng_feat.py 参照技术圈涵涵开源代码
9. 301_timediff_last_next_feat.py 时间差,ratio相关特征
10. 401_list_till_feat.py 本来是对三个list特征进行挖掘,复赛没有使用,而改为了一些新加的特征
11. 501_clickTran_feat.py 转化率特征,复赛放弃
12. 501_clickTran_feat_all.py 转化率特征,复赛放弃
13. 601_merge_data.py 合并之前构造的特征
14. lgb_feat_imp.csv 特征重要性

具体特征工程,构建思路,模型选择,数据下载参考知乎文章
>https://zhuanlan.zhihu.com/p/36858386

    path = './data/'

    data = pd.read_csv(path + 'all_final_data_11-07-18-51.csv')
    # data = pd.read_csv(path + '301_timediff_last_next_feat.csv')
    features, target = feat_select(data)
    ss = ShuffleSplit(len(data), test_size=0.25, random_state=0)
    df_train = data[features]
    df_target = data[target]
    for train, test in ss.split(df_train, df_target):
        train_data = df_train.values[train]
        train_target = df_target.values[train]
        test_data = df_train[test]
        test_target = df_target[test]
        best_iter = xgbCV(train_data, train_target, test_data, test_target)
        sub(train_data, train_target, test_data, test_target, best_iter)

