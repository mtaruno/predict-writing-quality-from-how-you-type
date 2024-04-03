# Predict Writing Quality from How You Type

Writing is fundamental to our success and self-improvement. From just the way you type and move your mouse, could we predict how well you will score on a writing test? This work explores the correlation between keystroke dynamics and essay writing quality, aiming to advance the capabilities of automated writing evaluation.

## Background

Our motivation of trying to see which behaviors are conducive to higher writing scores is statistically backed. Results indicated that keystroke indices accounted for 76% of the variance in essay quality and up to 38% of the variance in linguistic characteristics. In a regression between essay scores and keystroke indices, three variables combined to explain 76% of the variance in essay scores which is verbosity, largest latency, and backspaces. This means that it is reasonable to believe that just from the way you type and move your mouse during a writing exam, these behaviors are predictive of writing quality. This is interesting because in everybody’s journey of improving their writing quality or to find better writing exam strategies, seeing behaviors may have implications to encourage certain sets of behaviors over others. More practically, valuable insights may be provided for writing instruction, development of automated writing evaluation techniques, and intelligent tutoring systems.

### Definition

Each test is graded on an essay score received out of 6 (the prediction target for the competition). This target variable of test score is what we want to predict, and the competition is evaluated using the RMSE:

### Related Work

Keystroke analysis has been increasingly used to gain insight into the writing process. With keystroke logging, information on every key pressed and released is recorded, resulting in a detailed log on students' text composition.

Drawing from existing scholarly works, we categorize these features into five distinct groups:

1. Pause-related features encompass aspects like intervals between keystrokes and pauses preceding word initiation.
2. Revision-oriented features, which include metrics such as the frequency of backspace usage and the duration of editing pauses.
3. Verbosity-related features, chiefly represented by the total word count.
4. Fluency-related features, exemplified by the proportion of typing bursts culminating in revisions.
5. Features pertaining to non-character-producing keystrokes, encompassing actions like text selection, copy-paste operations, cut commands, and mouse navigations.

These categorizations aid in a comprehensive understanding and analysis of typing behaviors and patterns.

### Techniques

We can divide our techniques into two sections. The first is a data-side approach with feature engineering, creating different versions of the datasets. The second is a model-side approach. Kaggle is a competitive environment where minute insights which stem from experiments and exploratory data analysis (EDA) are the differentiator that lead to leader-board improvements.

For this competition dataset, in the data pipeline, we start with data cleaning. Then we divide into two parts: sentence reconstruction and feature engineering.

#### Finding the best essay reconstruction

Since scores are determined based on final text, this is a really important part of the competition. There were a lot of public discussion about specific reconstruction techniques that are effective. For example, there is an idea that if the cursor position and text change information don’t match with reconstructed text, you can search the sequence with the nearest fuzzy match. Also, it’s about recognizing when an Undo operation (ctrl+Z) was performed if cursor and text change information does not match with reconstructed text.

Once the basic reconstruction logic is implemented, a lot of feature engineering could be performed.

For our implementation, we created a feature called latency, which calculates the temporal differences in the keystroke event inputs.

- Longest: Finding the longest delay between consecutive keystrokes for each writer (longest)
- Smallest: shortest delay which represents the fastest typing speed
- Median value of delays: offering insight into typical typing rhythm
- Initial pause: time before first keystroke, possibly reflecting preparation time, and pause duration.
- Pause duration: Counts of pauses in different duration ranges (half a second, one second, etc.) which is useful for understanding the frequency and length of breaks in typing. In the pauses it might relate to cognitive processes like planning or revising text.

Ratio features were also created based on the keystroke data. This captures how much words were written in the given time and over the number of keystroke events, how many events are being done over the given time period, and the proportion of idle time in the typing period used.

### Data Cleaning

In the actual code pipeline not much data cleaning was employed, however based on our observations, there are key data cleaning steps that could have been taken. Namely this involves discarding events that strongly precede the first input, correcting up and down times to ensure gap times and action times are not too large, and fixing Unicode errors.

## Feature Engineering

For feature engineering, we use the keyboard activity data. The features (especially the custom built features) should aim to directly address the target feature of how the evaluator will score the essays.

- **Aggregations**: The aggregations included are quantiles, nunique, mean, standard deviation, skew, kurtosis, min/max, and sum. These aggregations seriously add to the number of features and are also applied to the time gap features. The total number of features reach 396 in our training dataset.
- **Time features**: Time features help capture the temporal nature and resolutions of the dataset. The gaps included are 1, 2, 3, 4, 10, 20, 50, and 100 event steps. The smaller gaps help capture immediate sequential dependencies while larger gaps help capture longer term patterns.
- **Counts**: Important counts include activity, event, and text change counts.

## Modelling

In terms of modelling, we tested three models, LightGBM, CatBoost, and SVM. Our fourth model is an ensemble of these three models.

- **LightGBM (Gradient Boosting)**: LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is known for its efficiency and speed, making it suitable for large datasets.
- **CatBoost (Categorical Boosting)**: CatBoost is another gradient boosting algorithm designed to handle categorical features efficiently. It is robust against overfitting and requires minimal hyperparameter tuning.
- **SVM (Support Vector Machines)**: Support Vector Machines are a class of supervised learning algorithms that can be used for regression tasks.

These models were chosen for their effectiveness in handling both numerical and categorical features, which are prevalent in keystroke and mouse movement data. We evaluate their performance using the root mean square error (RMSE), which aligns with the competition's evaluation metric.

The hyperparameters used for each of these models are summarized in the table below:

| Model | Hyperparameter | Value |
|-------|---------------|-------|
| LightGBM | num_leaves | 22 |
| | learning_rate | 0.0386 |
| | reg_alpha | 0.00768 |
| | reg_lambda | 0.342 |
| | colsample_bytree | 0.627 |
| | subsample | 0.855 |
| CatBoost | n_estimators | 12001 |
| | learning_rate | 0.03 |
| SVM | C | 1.0 |
| | kernel | 'rbf' |

Additionally, we ensemble (combining predictions from various models) these models to obtain a better result. We further tune the weights of the three models and selected the best one as our last parameters. We found the best blending weights to bet 0.459 for LGBM, 0.246 for CatBoost, and 0.295 for SVM.

## Experiments

In our experimental setup, we conducted individual tests for each of the three models: LightGBM, CatBoost, and SVM. Additionally, we performed an ensemble experiment combining these models. The experiments were designed to evaluate the models' performance in predicting essay scores based on the features extracted from keystroke and mouse movement data.

### Individual Model Experiments

For each model, we tuned hyperparameters as detailed in the Models section and trained them on a designated training set. We then validated their performance on a separate validation set.

### Ensemble Experiment

After individual testing, we combined the three models into an ensemble. The ensemble method aimed to leverage the strengths of each model to improve overall prediction accuracy. The final predictions were a weighted average of predictions from each model, with weights tuned to minimize the validation RMSE.

### Results

The results of our experiments are summarized in Figure below. The ensemble model demonstrated a notable improvement over individual models, indicating the effectiveness of combining different machine learning approaches for this task.

![Comparison of RMSE for individual models and the ensemble model.](1.png)

Since the leaderboard scores are fundamentally different from actual scores that will be on the private dataset, it is important to trust your own cross validation instead of fully rely on the leaderboard ranking. Our team ended up having a final RMSE of 0.56886 on the final leaderboard, which is 126 places higher than the initial leaderboard score with an RMSE of 0.570.

## Discussion

To further improve our results, it has been shown aligning context features using the reconstructed essays generated using a pre-trained Deberta-based regressor shows promising results. Deberta here essentially works to create a new feature that adds more relevant context to the feature set. Although GPU intensive, BERT and even LLaMA can be explored here. This general theme of introducing external data seems to work quite effectively. This external data can be anonymized, have a vectorizer (such as Tf-idf applied), and then essay related features can be merged together with the competition data reconstructed essays. Choosing the optimal feature set and performing dimensionality reduction with SVD, PCA, or perhaps in a more advanced way with regression feature importances also seem to be effective. This has been a fruitful project, shedding light on the depths that could be taken by the Kaggle community to create relevant features collaboratively and showcasing the power of data science in creating automated grading scheme tools and discovering just how observing something as general as keystrokes can powerfully predict writing quality.
