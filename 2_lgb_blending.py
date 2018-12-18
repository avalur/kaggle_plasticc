import pandas as pd
import numpy as np

lgb_1036 = pd.read_csv('single_subm_0.118646_0.872081_2018-12-17-05-26.csv')
all_1021 = pd.read_csv('single_subm_pred_extragalactic_gen_unk.csv')

res = 0.1*lgb_1036 + 0.9*all_1021
res['object_id'] = res['object_id'].astype(np.int64)
res.to_csv('01_lgb_1036_09_all_1021.csv', index=False)
