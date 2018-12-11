import sys
import pandas as pd
import numpy as np

from competition_spec_loss import multi_weighted_logloss

classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}


def main(argc, argv):
    a = pd.read_csv(argv[1])
    b = pd.read_csv(argv[2])

    meta_train = pd.read_csv('../input/training_set_metadata.csv')
    # train = pd.read_csv('../input/training_set.csv')
    y_true = meta_train['target']

    for p in np.linspace(0, 1, 21):
        res = p*a + (1 - p)*b
        del res['object_id']
        print(multi_weighted_logloss(y_true, res.values, classes, class_weights))


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
