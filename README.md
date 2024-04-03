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
