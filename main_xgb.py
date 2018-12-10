import pandas as pd
import sys

from competition_spec_utils import process_meta, featurize
from datetime import datetime
from functools import partial
from xgb_cross_validation import xgb_modeling_cross_validation
from predict import process_test

xgb_params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'silent': True,
    'num_class': 14,

    'booster': 'gbtree',
    'n_jobs': 4,
    'n_estimators': 1000,
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'base_score': 0.25,
    'max_depth': 7,
    'max_delta_step': 2,  # default=0
    'learning_rate': 0.03,
    'max_leaves': 11,
    'min_child_weight': 64,
    'gamma': 0.1,  # default=
    'subsample': 0.7,
    'colsample_bytree': 0.68,
    'reg_alpha': 0.01,  # default=0
    'reg_lambda': 10.,  # default=1
    'seed': 537

}

lgbm_params = {
    'device': 'cpu',
    'objective': 'multiclass',
    'num_class': 14,
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'max_depth': 7,
    'n_estimators': 500,
    'subsample_freq': 2,
    'subsample_for_bin': 5000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 1.0,
    'cat_smooth': 59.5,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.0267,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.1,
    'num_leaves': 7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.00023,
    'skip_drop': 0.44,
    'subsample': 0.75
}


def main(argc, argv):
    meta_train = process_meta('../input/training_set_metadata.csv')

    train = pd.read_csv('../input/training_set.csv')
    full_train = featurize(train, meta_train)

    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']

    classes = sorted(y.unique())
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})
    print('Unique classes : {}, {}'.format(len(classes), classes))
    print(class_weights)
    # sanity check: classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    # sanity check: class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2,
    #                                65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

    if 'object_id' in full_train:
        object_id = full_train['object_id']
        del full_train['object_id']
        del full_train['hostgal_specz']
        del full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b']
        del full_train['ddf']

    train_mean = full_train.mean(axis=0)
    # train_mean.to_hdf('train_data.hdf5', 'data')
    pd.set_option('display.max_rows', 500)
    print(full_train.describe().T)
    full_train.fillna(0, inplace=True)

    eval_func = partial(xgb_modeling_cross_validation,
                        full_train=full_train,
                        y=y,
                        classes=classes,
                        class_weights=class_weights,
                        id=object_id,
                        nr_fold=5,
                        random_state=1,
                        )

    # modeling from CV
    clfs, score = eval_func(xgb_params)

    filename = 'subm_{:.6f}_{}.csv'.format(score, datetime.now().strftime('%Y-%m-%d-%H-%M'))
    print('save to {}'.format(filename))

    # TEST
    process_test(clfs,
                 features=full_train.columns,
                 train_mean=train_mean,
                 filename=filename,
                 chunks=5000000)

    z = pd.read_csv(filename)
    print("Shape BEFORE grouping: {}".format(z.shape))
    z = z.groupby('object_id').mean()
    print("Shape AFTER grouping: {}".format(z.shape))
    z.to_csv('single_{}'.format(filename), index=True)


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
