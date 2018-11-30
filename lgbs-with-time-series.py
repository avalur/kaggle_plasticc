"""
PLAsTiCC_in_a_kernel_meta_and_data
----------------------------------
First version from https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data

@website https://www.kaggle.com/aleksandravdyushenko
@author Aleksandr Avdyushenko https://www.kaggle.com/aleksandravdyushenko
"""
import gc
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


def multi_weighted_logloss(y_true, y_preds):
    """
    @author Aleksandr Avdyushenko https://www.kaggle.com/aleksandravdyushenko
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {x: 1 for x in classes}
    class_weight[15] = 2
    class_weight[64] = 2

    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true by one-hot-encoding
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class_99 for now, since there is no class_99 in the training set
    # So we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_weight_arr = np.array([class_weight[k]
                                 for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_weight_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_weight_arr)
    return loss


def lgb_multi_weighted_logloss(y_true, y_preds):
    loss = multi_weighted_logloss(y_true, y_preds)
    return 'wloss', loss, False


gc.enable()

# '../input/'
data_dir = '/data/ajavdjushenko'
train = pd.read_csv(os.path.join(data_dir, 'training_set.csv'))


def apply_aggs(data):
    aggs = {
        'mjd': ['min', 'max', 'size'],
        # 'passband': ['min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std'],
        'detected': ['min', 'max', 'mean', 'median', 'std'],
    }

    by_columns = ['object_id', 'passband']
    agg_data = data.groupby(by_columns, as_index=False).agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_data.columns = by_columns + new_columns
    agg_data['mjd_diff'] = agg_data['mjd_max'] - agg_data['mjd_min']
    del agg_data['mjd_max'], agg_data['mjd_min']
    agg_data = agg_data.set_index('object_id')
    return agg_data


agg_train = apply_aggs(train)
del train
gc.collect()

meta_train = pd.read_csv(os.path.join(data_dir, 'training_set_metadata.csv'))
# meta_train.head()

full_train = agg_train.reset_index().merge(
    right=meta_train,
    how='outer',
    on='object_id'
)

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']
classes = sorted(y.unique())

# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes: ', classes)

if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'], full_train['distmod']
    del full_train['hostgal_specz']

# Filling na
train_mean = full_train.mean(axis=0)
full_train.fillna(train_mean, inplace=True)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
clfs = []
importances = pd.DataFrame()

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'learning_rate': 0.03,
    'subsample': .9,
    'colsample_bytree': .7,
    'reg_alpha': .01,
    'reg_lambda': .01,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'n_estimators': 1000,
    'silent': -1,
    'max_depth': 3
}

oof_preds = np.zeros((len(full_train), len(classes)))
for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
    val_x, val_y = full_train.iloc[val_], y.iloc[val_]

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        eval_metric=lgb_multi_weighted_logloss,
        verbose=100,
        early_stopping_rounds=50
    )
    oof_preds[val_, :] = clf.predict_proba(
        val_x, num_iteration=clf.best_iteration_)
    print(multi_weighted_logloss(val_y, clf.predict_proba(
        val_x, num_iteration=clf.best_iteration_)))

    imp_df = pd.DataFrame()
    imp_df['feature'] = full_train.columns
    imp_df['gain'] = clf.feature_importances_
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0)

    clfs.append(clf)

print('MULTI WEIGHTED LOG LOSS : %.5f ' %
      multi_weighted_logloss(y_true=y, y_preds=oof_preds))

mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

# plt.figure(figsize=(8, 12))
# sns.barplot(x='gain', y='feature', data=importances.sort_values('mean_gain', ascending=False))
# plt.tight_layout()
# plt.savefig('importances.png')

# test predict

meta_test = pd.read_csv(os.path.join(data_dir, 'test_set_metadata.csv'))

import time

start = time.time()
chunks = 5000000

test_path = os.path.join(data_dir, 'test_set.csv.zip')

for i_c, df in enumerate(pd.read_csv(test_path,
                                     chunksize=chunks, iterator=True)):
    # Group by object id and passband
    agg_test = apply_aggs(df)

    del df
    gc.collect()

    # Merge with meta data
    full_test = agg_test.reset_index().merge(
        right=meta_test,
        how='left',
        on='object_id'
    )

    # Filling na from train
    full_test = full_test.fillna(train_mean)

    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(
                full_test[full_train.columns]) / folds.n_splits
        else:
            preds += clf.predict_proba(
                full_test[full_train.columns]) / folds.n_splits

    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])

    # Store predictions
    preds_df = pd.DataFrame(
        preds, columns=['class_' + str(s) for s in clfs[0].classes_])
    preds_df['object_id'] = full_test['object_id']
    preds_df['class_99'] = preds_99

    if i_c == 0:
        preds_df.to_csv('predictions.csv', header=True, mode='w', index=False)
    else:
        preds_df.to_csv('predictions.csv', header=False, mode='a', index=False)

    del agg_test, full_test, preds_df, preds
    gc.collect()

    if (i_c + 1) % 2 == 0:
        print('%15d done in %5.1f' %
              (chunks * (i_c + 1), (time.time() - start) / 60))

# Store single predictions
z = pd.read_csv('predictions.csv')

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

z.to_csv('single_predictions.csv', index=True)
