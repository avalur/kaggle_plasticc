import numpy as np
import pandas as pd
from functools import partial
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

from competition_spec_loss import xgb_multi_weighted_logloss, multi_weighted_logloss


def xgb_modeling_cross_validation(params,
                                  full_train,
                                  y,
                                  classes,
                                  class_weights,
                                  id,
                                  nr_fold=5,
                                  random_state=1,
                                  ):
    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))

    # loss function
    func_loss = partial(xgb_multi_weighted_logloss,
                        classes=classes,
                        class_weights=class_weights)

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        clf = XGBClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=func_loss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)
        print('no {}-fold loss: {}'.format(fold_ + 1,
                                           multi_weighted_logloss(val_y, oof_preds[val_, :],
                                                                  classes, class_weights)))

        imp_df = pd.DataFrame({
            'feature': full_train.columns,
            'gain': clf.feature_importances_,
            'fold': [fold_ + 1] * len(full_train.columns),
        })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds,
                                   classes=classes, class_weights=class_weights)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))

    oof_preds_pd = pd.DataFrame(data=oof_preds, columns=['class_{}'.format(s) for s in classes])
    pd.concat([id, oof_preds_pd], axis=1).to_csv('xgb_train_oof_preds.csv', index=False)
    df_importances = save_importances(importances_=importances)
    df_importances.to_csv('xgb_importances.csv', index=False)

    return clfs, score

def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_
