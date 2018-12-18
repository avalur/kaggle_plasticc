import sys
import pandas as pd


def gen_unknown(data):
    return (0.5 + 0.5 * data["mymedian"] + 0.25 * data["mymean"] - 0.5 * data["mymax"] ** 3) / 2


feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

galactic_classes = [6, 16, 53, 65, 92]
extragalactic_classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]


def class_99(subm):
    y = pd.DataFrame()
    y['mymean'] = subm[feats].mean(axis=1)
    y['mymedian'] = subm[feats].median(axis=1)
    y['mymax'] = subm[feats].max(axis=1)

    subm['class_99'] = gen_unknown(y)
    subm.to_csv('single_subm_2_lgbm_gen_unknown.csv', index=False)


def extragalactic(subm):
    meta_test = pd.read_csv('../input/test_set_metadata.csv')
    result = subm.merge(right=meta_test, on='object_id')

    galactic_cut = meta_test['hostgal_photoz'] == 0

    galactic_classes = ['class_{}'.format(s) for s in [6, 16, 53, 65, 92]]
    extragalactic_classes = ['class_{}'.format(s) for s in [15, 42, 52, 62, 64, 67, 88, 90, 95]]
    result.loc[galactic_cut, extragalactic_classes] = 0.0
    result.loc[~galactic_cut, galactic_classes] = 0.0

    classes_all = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
    columns = ['object_id'] + ['class_{}'.format(s) for s in classes_all]
    result[columns].to_csv('single_subm_pred_extragalactic_gen_unk.csv', index=False)


def main(argc, argv):
    subm = pd.read_csv('single_subm_0.140558_0.878743_2018-12-15-09-51.csv')
    class_99(subm)
    # extragalactic(subm)


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
