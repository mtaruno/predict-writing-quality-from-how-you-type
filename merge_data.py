import pandas as pd


INPUT_DIR = './data'
ARTIFACTS_DIR = './artifacts'

train_feats = pd.read_csv(f'{ARTIFACTS_DIR}/train_feats.csv')
test_feats = pd.read_csv(f'{ARTIFACTS_DIR}/test_feats.csv')



train_feats = train_feats.merge(train_agg_fe_df, on='id', how='left')
test_feats = test_feats.merge(test_agg_fe_df, on='id', how='left')