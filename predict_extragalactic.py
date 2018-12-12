import gc; gc.enable()
import numpy as np
import pandas as pd
import time

from competition_spec_utils import process_meta, featurize

galactic_classes = [6, 16, 53, 65, 92]
extragalactic_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]


def predict_chunk(df_, clfs_gal_, clfs_ext_, meta_, features):
    # process all features
    full_test = featurize(df_, meta_)
    full_test.fillna(0, inplace=True)

    galactic_cut = full_test['hostgal_photoz'] == 0
    gal_test = full_test[galactic_cut]
    ext_test = full_test[~galactic_cut]

    # Make predictions
    preds_gal = None
    for clf in clfs_gal_:
        if preds_gal is None:
            preds_gal = clf.predict_proba(gal_test[features])
        else:
            preds_gal += clf.predict_proba(gal_test[features])

    preds_gal = preds_gal / len(clfs_gal_)

    preds_99_gal = np.ones(preds_gal.shape[0])
    for i in range(preds_gal.shape[1]):
        preds_99_gal *= (1 - preds_gal[:, i])

    # Create DataFrame from predictions
    preds_gal = pd.DataFrame(preds_gal,
                             columns=['class_{}'.format(s) for s in clfs_gal_[0].classes_])
    preds_gal['object_id'] = gal_test['object_id']
    for c in ['class_{}'.format(s) for s in extragalactic_classes]:
        preds_gal.insert(0, c, 0.0)
    preds_gal['class_99'] = 0.017 * preds_99_gal / np.mean(preds_99_gal)

    preds_ext = None
    for clf in clfs_ext_:
        if preds_ext is None:
            preds_ext = clf.predict_proba(ext_test[features])
        else:
            preds_ext += clf.predict_proba(ext_test[features])

    preds_ext = preds_ext / len(clfs_ext_)

    preds_99_ext = np.ones(preds_ext.shape[0])
    for i in range(preds_ext.shape[1]):
        preds_99_ext *= (1 - preds_ext[:, i])

    # Create DataFrame from predictions
    preds_ext = pd.DataFrame(preds_ext,
                             columns=['class_{}'.format(s) for s in clfs_ext_[0].classes_])
    preds_ext['object_id'] = ext_test['object_id']
    for c in ['class_{}'.format(s) for s in galactic_classes]:
        preds_ext.insert(0, c, 0.0)
    preds_ext['class_99'] = 0.17 * preds_99_ext / np.mean(preds_99_ext)

    preds_df_ = pd.concat([preds_gal, preds_ext])
    return preds_df_


def process_test(clfs_gal, clfs_ext,
                 features,
                 filename='predictions.csv',
                 chunks=5000000):
    start = time.time()

    meta_test = process_meta('../input/test_set_metadata.csv')
    # meta_test.set_index('object_id',inplace=True)

    remain_df = None
    for i_c, df in enumerate(pd.read_csv('../input/test_set.csv.zip', chunksize=chunks, iterator=True)):
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])

        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df

        preds_df = predict_chunk(df_=df,
                                 clfs_gal_=clfs_gal,
                                 clfs_ext_=clfs_ext,
                                 meta_=meta_test,
                                 features=features)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes'.format(
            chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    # Compute last object in remain_df
    preds_df = predict_chunk(df_=remain_df,
                             clfs_gal_=clfs_gal,
                             clfs_ext_=clfs_ext,
                             meta_=meta_test,
                             features=features)

    preds_df.to_csv(filename, header=False, mode='a', index=False)
    return
