import pandas as pd
from utils import getEssays, q1, q3, compute_paragraph_aggregations, compute_sentence_aggregations, split_essays_into_sentences, split_essays_into_paragraphs


INPUT_DIR = './data'
ARTIFACTS_DIR = './artifacts'


train_logs = pd.read_csv(f'{INPUT_DIR}/train_logs.csv')
train_scores = pd.read_csv(f'{INPUT_DIR}/train_scores.csv')
test_logs = pd.read_csv(f'{INPUT_DIR}/test_logs.csv')
ss_df = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
train_essays = pd.read_csv(f'{INPUT_DIR}/train_essays_02.csv')
train_essays.index = train_essays["Unnamed: 0"]
train_essays.index.name = None
train_essays.drop(columns=["Unnamed: 0"], inplace=True)
train_essays.head()

print("Creating sentence features for train dataset train_sent_agg_df.csv...")

# Sentence features for train dataset
train_sent_df = split_essays_into_sentences(train_essays)
train_sent_agg_df = compute_sentence_aggregations(train_sent_df)
train_sent_agg_df.to_csv(ARTIFACTS_DIR + '/train_sent_agg_df.csv', index=False)


print("Creating train_paragraph_agg_df.csv...")

# Paragraph features for train dataset
train_paragraph_df = split_essays_into_paragraphs(train_essays)
train_paragraph_agg_df = compute_paragraph_aggregations(train_paragraph_df)
train_paragraph_agg_df.to_csv(ARTIFACTS_DIR + '/train_paragraph_agg_df.csv', index=False)


print("Creating test_paragraph_agg_df.csv...")

# Features for test dataset
test_essays = getEssays(test_logs)
test_sent_agg_df = compute_sentence_aggregations(split_essays_into_sentences(test_essays))
test_paragraph_agg_df = compute_paragraph_aggregations(split_essays_into_paragraphs(test_essays))
test_paragraph_agg_df.to_csv(ARTIFACTS_DIR + '/test_paragraph_agg_df.csv', index=False)


# train_logs.to_csv(ARTIFACTS_DIR + "/train_logs.csv", index=False)
# train_scores.to_csv(ARTIFACTS_DIR + "/train_scores.csv", index=False)
# test_logs.to_csv(ARTIFACTS_DIR + "/test_logs.csv", index=False)
# ss_df.to_csv(ARTIFACTS_DIR +  "/sample_submission.csv", index=False)

