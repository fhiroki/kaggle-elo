from datetime import datetime
import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import utils

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min',
                  'hist_card_id_size', 'new_purchase_date_max',
                  'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df, test_df, target, num_folds, stratified=False, debug=False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params = {
                'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.003,
                'subsample': 0.9855232997390695,
                'max_depth': 8,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'device': 'cpu' if debug else 'gpu',
                'n_jobs': -1,
                'seed': int(2**n_fold),
                'bagging_seed': int(2**n_fold),
                'drop_seed': int(2**n_fold)
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds=100,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain',
                                                                           iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f\n' % (n_fold + 1, utils.rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    score = utils.rmse(target, oof_preds)
    print('Final RMSE : %.6f\n' % (score))

    # display importances
    utils.display_importances(feature_importance_df)

    if not debug:
        # save submission file
        test_df.loc[:, 'target'] = sub_preds
        test_df = test_df.reset_index()
        combine_db = pd.read_csv('../input/combining_submission.csv')
        test_df['target'] = 0.6 * test_df['target'] + 0.4 * combine_db['target']

        filename = 'score_{:.6f}_{}.csv'.format(score, datetime.now().strftime('%Y-%m-%d-%H-%M'))
        test_df[['card_id', 'target']].to_csv(os.path.join("../output", filename), index=False)
