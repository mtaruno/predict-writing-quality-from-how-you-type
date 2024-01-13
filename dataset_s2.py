
import pandas as pd

INPUT_DIR = './data'
ARTIFACTS_DIR = './artifacts'


train_logs = pd.read_csv(INPUT_DIR+"/train_logs.csv")
test_logs = pd.read_csv(INPUT_DIR+"/test_logs.csv")

train_logs = pd.read_csv(INPUT_DIR+"/train_logs.csv")
test_logs = pd.read_csv(INPUT_DIR+"/test_logs.csv")

print("Creating train_agg_fe_df.csv...")
train_agg_fe_df = train_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
    ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
train_agg_fe_df.reset_index(inplace=True)

train_agg_fe_df.to_csv(ARTIFACTS_DIR+"/train_agg_fe.csv", index=False)

print("Creating test_agg_fe_df.csv...")
test_agg_fe_df = test_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
    ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
test_agg_fe_df.reset_index(inplace=True)

test_agg_fe_df.to_csv(ARTIFACTS_DIR+"/test_agg_fe.csv", index=False)


# Code for creating these features comes from here: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
# Idea is based on features introduced in Section 3 of this research paper: https://files.eric.ed.gov/fulltext/ED592674.pdf

data = []

for logs in [train_logs, test_logs]:
    logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])
    logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged']) / 1000

    group = logs.groupby('id')['time_diff']
    largest_lantency = group.max()
    smallest_lantency = group.min()
    median_lantency = group.median()
    initial_pause = logs.groupby('id')['down_time'].first() / 1000
    pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
    pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
    pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
    pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
    pauses_3_sec = group.apply(lambda x: (x > 3).sum())

    data.append(pd.DataFrame({
        'id': logs['id'].unique(),
        'largest_lantency': largest_lantency,
        'smallest_lantency': smallest_lantency,
        'median_lantency': median_lantency,
        'initial_pause': initial_pause,
        'pauses_half_sec': pauses_half_sec,
        'pauses_1_sec': pauses_1_sec,
        'pauses_1_half_sec': pauses_1_half_sec,
        'pauses_2_sec': pauses_2_sec,
        'pauses_3_sec': pauses_3_sec,
    }).reset_index(drop=True))

train_eD592674, test_eD592674 = data

train_feats = pd.read_csv(ARTIFACTS_DIR+'/train_feats.csv')
test_feats = pd.read_csv(ARTIFACTS_DIR+'/test_feats.csv')

train_feats = train_feats.merge(train_eD592674, on='id', how='left')
test_feats = test_feats.merge(test_eD592674, on='id', how='left')

train_scores = pd.read_csv(INPUT_DIR+'/train_scores.csv')

train_feats = train_feats.merge(train_scores, on='id', how='left')

train_feats.to_csv(ARTIFACTS_DIR+'/train_feats_v2.csv', index=False)
test_feats.to_csv(ARTIFACTS_DIR+'/test_feats_v2.csv', index=False)


