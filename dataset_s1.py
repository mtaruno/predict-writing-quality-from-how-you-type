from preprocessor import Preprocessor
import pandas as pd


INPUT_DIR = './data'
ARTIFACTS_DIR = './artifacts'

# read in train_logs.csv
train_logs = pd.read_csv(INPUT_DIR+"/train_logs.csv")
test_logs = pd.read_csv(INPUT_DIR+"/test_logs.csv")

preprocessor = Preprocessor(seed=42)
print("Applying preprocessor on train_logs.csv")
train_feats = preprocessor.make_feats(train_logs)
train_feats.to_csv(ARTIFACTS_DIR + '/train_feats.csv', index=False)

print("Applying preprocessor on test_logs.csv")
test_feats = preprocessor.make_feats(test_logs)
nan_cols = train_feats.columns[train_feats.isna().any()].tolist()
train_feats = train_feats.drop(columns=nan_cols)
test_feats = test_feats.drop(columns=nan_cols)
test_feats.to_csv(ARTIFACTS_DIR + '/test_feats.csv', index=False)