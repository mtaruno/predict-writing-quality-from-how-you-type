import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics

# 假设 train_feats 和 test_feats 已经被正确加载和预处理
# train_cols, target_col 需要被正确定义

models_dict = {}
scores = []

file_train_path = 'train_feats.csv'
# 读取CSV文件
train_feats = pd.read_csv(file_train_path).drop("Unnamed: 0", axis=1)
test_feats = pd.read_csv('test_feats.csv').drop("Unnamed: 0", axis=1)
# 获取形状
shape = train_feats.shape
# 打印形状
print(f"Rows: {shape[0]}, Columns: {shape[1]}")
print(f"Test Rows: {test_feats.shape[0]}, Columns: {test_feats.shape[1]}")

target_col = ['score']
drop_cols = ['id']
train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]
# X_train = train_feats[train_cols]
# print(X_train.shape)

# X_test = test_feats[train_cols]
# print(X_test.shape)
# print(X_test)

# 检查并替换无穷大值
# train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
# train_feats.fillna(train_feats.mean(numeric_only=True), inplace=True)

def check_special_json_characters_in_column_names(df):
    special_characters = '["{}[]()<>?@!#%^&*-=+~`|\\/\':;,]'
    problematic_columns = [col for col in df.columns if any(char in col for char in special_characters)]
    
    if problematic_columns:
        print("Columns with special JSON characters:")
        for col in problematic_columns:
            print(col)
    else:
        print("No columns with special JSON characters found.")

def remove_special_json_characters(df):
    special_characters = '["{}[]()<>?@!#%^&*-=+~`|\\/\':;,]'
    df.columns = df.columns.str.replace(special_characters, '', regex=True)
    return df

# 使用该函数清理train_feats和test_feats的列名
train_feats = remove_special_json_characters(train_feats)
test_feats = remove_special_json_characters(test_feats)

# # 使用该函数检查train_feats和test_feats的列名
check_special_json_characters_in_column_names(train_feats)
check_special_json_characters_in_column_names(test_feats)

# 初始化 OOF 和 TEST 预测数组
OOF_PREDS = np.zeros((train_feats.shape[0], 1))
TEST_PREDS = np.zeros((test_feats.shape[0], 1))

best_params = {
    'reg_alpha': 0.007678095440286993, 
    'reg_lambda': 0.34230534302168353, 
    'colsample_bytree': 0.627061253588415, 
    'subsample': 0.854942238828458, 
    'learning_rate': 0.038697981947473245, 
    'num_leaves': 22, 
    'max_depth': 37, 
    'min_child_samples': 18,
    'n_jobs': 4
}


# Code comes from here: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs

models_dict = {}
scores = []

test_predict_list = []
best_params = {'reg_alpha': 0.007678095440286993, 
               'reg_lambda': 0.34230534302168353, 
               'colsample_bytree': 0.627061253588415, 
               'subsample': 0.854942238828458, 
               'learning_rate': 0.038697981947473245, 
               'num_leaves': 22, 
               'max_depth': 37, 
               'min_child_samples': 18,
               'n_jobs':4
              }

for i in range(5): 
    kf = model_selection.KFold(n_splits=10, random_state=42 + i, shuffle=True)
    oof_valid_preds = np.zeros(train_feats.shape[0])
    X_test = test_feats[train_cols]
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
        
        X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
        X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]
        params = {
            "objective": "regression",
            "metric": "rmse",
            'random_state': 42,
            "n_estimators" : 12001,
            "verbosity": -1,
            **best_params
        }
        model = lgb.LGBMRegressor(**params)
        early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
        
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
                  callbacks=[early_stopping_callback],
        )
        valid_predict = model.predict(X_valid)
        oof_valid_preds[valid_idx] = valid_predict
        OOF_PREDS[valid_idx, 0] += valid_predict / 5
        test_predict = model.predict(X_test)
        TEST_PREDS[:, 0] += test_predict / 5 / 10
        test_predict_list.append(test_predict)
        score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
        models_dict[f'{fold}_{i}'] = model

    oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
    scores.append(oof_score)

print('OOF metric LGBM = {:.5f}'.format(metrics.mean_squared_error(train_feats[target_col], 
                                                                   OOF_PREDS[:, 0], 
                                                                   squared=False)))
# for i in range(5): 
#     kf = model_selection.KFold(n_splits=10, random_state=42 + i, shuffle=True)
#     oof_valid_preds = np.zeros(train_feats.shape[0])
#     X_test = test_feats[train_cols]
    
#     for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
#         X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col].values.ravel()
#         X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col].values.ravel()
        
#         params = {
#             "objective": "regression",
#             "metric": "rmse",
#             'random_state': 42,
#             "n_estimators": 12001,
#             "verbosity": -1,
#             **best_params
#         }
        
#         model = lgb.LGBMRegressor(**params)
#         early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
        
#         model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping_callback])
#         valid_predict = model.predict(X_valid)
#         oof_valid_preds[valid_idx] = valid_predict
#         OOF_PREDS[valid_idx, 0] += valid_predict / 5
#         test_predict = model.predict(X_test)
#         TEST_PREDS[:, 0] += test_predict / 5 / 10
#         models_dict[f'{fold}_{i}'] = model

#         score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
#         models_dict[f'{fold}_{i}'] = model

#     oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
#     scores.append(oof_score)

# print('OOF metric LGBM = {:.5f}'.format(np.mean(scores)))
