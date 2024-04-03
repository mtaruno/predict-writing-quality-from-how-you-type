# predict-writing-quality-from-how-you-type

Writing is fundamental to our success and self-improvement. From just the way you type and move your mouse, could we predict how well you will score on a writing test? Here is our GitHub repository where extensive testing is done: \href{https://github.com/mtaruno/predict-writing-quality-from-how-you-type}{https://github.com/mtaruno/predict-writing-quality-from-how-you-type}. 
This work explores the correlation between keystroke dynamics and essay writing quality, aiming to advance the capabilities of automated writing evaluation. Our methodology involves two stages. In the data-centric phase, we focus on meticulous essay reconstruction from keystroke data from which many important features such as latency and ratio metrics are developed. Concurrently, we engage in comprehensive feature engineering, utilizing keyboard activity data to develop predictors directly aligned with essay evaluation criteria. Our modeling phase evaluates three distinct algorithms: LightGBM, CatBoost, and SVM, each chosen for their efficacy in processing diverse data types inherent in keystroke and mouse movement data. These models are put into an ensemble aiming to harness their collective strengths for improved predictive accuracy, as evidenced by a significant reduction in RMSE. This work sheds light on the intricate relationship between typing behavior and writing quality and this robust predictive framework offers valuable insights for educational tools and automated writing assessment systems, paving the way for more nuanced and effective writing evaluation methodologies.


### Background
Our motivation of trying to see which behaviors are conducive to higher writing scores is statistically backed. Results indicated that keystroke indices accounted for 76\% of the variance in essay quality and up to 38\% of the variance in linguistic characteristics. In a regression between essay scores and keystroke indices, three variables combined to explain 76\% of the variance in essay scores which is verbosity, largest latency, and backspaces. This means that it is reasonable to believe that just from the way you type and move your mouse during a writing exam, these behaviors are predictive of writing quality. This is interesting because in everybody’s journey of improving their writing quality or to find better writing exam strategies, seeing behaviors may have implications to encourage certain sets of behaviors over others. More practically, valuable insights may be provided for writing instruction, development of automated writing evaluation techniques, and intelligent tutoring systems. 

### Definition

Each test is graded on an essay score received out of 6 (the prediction target for the competition). This target variable of test score is what we want to predict, and the competition is evaluated using the RMSE: $RMSE = (\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2)^{1/2}$

Alternatively, there is a prize that rewards model efficiency because highly accurate models tend to be computationally heavy and leave a stronger carbon footprint. These models would like to be used to help educational organizations with limited computational capacities, so here both runtime and predictive performance is measured: $Efficiency = \frac{RMSE}{Base - minRMSE} + \frac{RuntimeSeconds}{32400}$

### Related Work
% Some research works have been done~\cite{conijn2022early}.
Keystroke analysis has been increasingly used to gain insight into the writing process. With keystroke logging, information on every key pressed and released is recorded, resulting in a detailed log on students' text composition. \citep{leijten2013keystroke, lindgren2019observing} 

Drawing from existing scholarly works, we categorize these features into five distinct groups: (1) Pause-related features encompass aspects like intervals between keystrokes and pauses preceding word initiation, as highlighted in Barkaoui's 2016 study~\citep{barkaoui2016and} and in Medimorec \& Risko study~\citep{medimorec2017pauses}; (2) Revision-oriented features, which include metrics such as the frequency of backspace usage and the duration of editing pauses\cite{deane2014using}; (3) Verbosity-related features, chiefly represented by the total word count \cite{likens2017keystroke}; (4) Fluency-related features, exemplified by the proportion of typing bursts culminating in revisions\cite{baaijen2012keystroke}; and (5) Features pertaining to non-character-producing keystrokes, encompassing actions like text selection, copy-paste operations, cut commands, and mouse navigations. These categorizations aid in a comprehensive understanding and analysis of typing behaviors and patterns.\cite{leijten2019mapping}

Machine learning method was introduced to predict final grade and classify students who might need support at several points during the writing process but results are even worse than baselines, which seem to point out that the relationship between keystroke data and writing quality might be less clear than previously posited. ~\citep{conijn2022early} 



### Techniques

We can divide our techniques into two sections. The first is a data-side approach with feature engineering, creating different versions of the datasets. The second is a model-side approach. Kaggle is a competitive environment where minute insights which stem from experiments and exploratory data analysis (EDA) are the differentiator that lead to leader-board improvements. For this competition dataset, in the data pipeline, we start with data cleaning. Then we divide into two parts: sentence reconstruction and feature engineering. 


\subsection{Finding the best essay reconstruction}
Since scores are determined based on final text, this is a really important part of the competition. There were a lot of public discussion about specific reconstruction techniques that are effective. For example, there is an idea that if the cursor position and text change information don’t match with reconstructed text, you can search the sequence with the nearest fuzzy match. Also, it’s about recognizing when an Undo operation (ctrl+Z) was performed if cursor and text change information does not match with reconstructed text. Once the basic reconstruction logic is implemented, a lot of feature engineering could be performed. 

For our implementation, we created a feature called latency, which calculates the temporal differences in the keystroke event inputs. . 
\begin{itemize}
    \item Longest: Finding the longest delay between consecutive keystrokes for each writer (longest)
    \item Smallest: shortest delay which represents the fastest typing speed
    \item Median value of delays: offering insight into typical typing rhythm)
    \item Initial pause: time before first keystroke, possibly reflecting preparation time, and pause duration. 
    \item Pause duration: Counts of pauses in different duration ranges (half a second, one second, etc.) which is useful for understanding the frequency and length of breaks in typing. In the pauses it might relate to cognitive processes like planning or revising text. 
\end{itemize}

Ratio features were also created based on the keystroke data. This captures how much words were written in the given time and over the number of keystroke events, how many events are being done over the given time period, and the proportion of idle time in the typing period used. 

### Data Cleaning
In the actual code pipeline not much data cleaning was employed, however based on our observations, there are key data cleaning steps that could have been taken. Namely this involves discarding events that strongly precede the first input, correcting up and down times to ensure gap times and action times are not too large, and fixing Unicode errors. 
