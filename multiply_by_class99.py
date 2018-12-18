import pandas as pd
import numpy as np

blend = pd.read_csv('05_lgb_1036_05_all_1021.csv')

class99 = blend['class_99']
id = blend['object_id']
blend = blend.multiply(class99, axis="index")
blend['class_99'] = class99.values
blend['object_id'] = id.values
res = blend
res['object_id'] = res['object_id'].astype(np.int64)
res.to_csv('05_lgb_1036_05_all_1021_multiply99.csv', index=False)
