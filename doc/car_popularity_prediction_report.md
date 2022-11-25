Car Popularity Prediction Report
================
Lisa Sequeira, Alex Taciuk, Peng Zhang, Gaoxiang Wang
2022-11-25

-   <a href="#summary" id="toc-summary">Summary</a>
-   <a href="#introduction" id="toc-introduction">Introduction</a>
-   <a href="#methods" id="toc-methods">Methods</a>
    -   <a href="#data" id="toc-data">Data</a>
    -   <a href="#question-of-interest" id="toc-question-of-interest">Question
        of Interest</a>
    -   <a href="#analysis" id="toc-analysis">Analysis</a>
-   <a href="#results--discussion" id="toc-results--discussion">Results
    &amp; Discussion</a>
-   <a href="#references" id="toc-references">References</a>

## Summary

Expecting a growing demand on used vehicles in the next years, we are
interested to build a classification machine learning model to predict
the popularity of a given used cars. The training data was from a public
available 1990s Car Evaluation dataset with 1728 observations, 6
categorical features and 1 categorical target. Data reading, data
splitting, and Exploratory Data Analysis (EDA) in the Python environment
have been performed. After that, `OrdinalEncoder` transformer has been
applied to pre-process the 6 features, then we were using four different
scikit-learn classifiers, `DummyClassifier`, `DecisionTreeClassifier`,
`RandomForestClassifier`, and `MultinomialNB` to conduct classifier
tuning by using `balanced_accuracy` as the scoring metric. The result
shown `RandomForestClassifier` had the best test score so we continued
to complete a hyperparameter optimization on the selected
`RandomForestClassifier`. At the end of the analysis, our best optimized
classifier has been applied on the test data. We received a optimistic
test score with 0.965.

## Introduction

According to Environment and Climate Change Canada \[1\], the federal
government announced “by 2035, all new cars and light-duty trucks sold
in the country will be zero-emission vehicles.” This was really an
exciting news from tacking climate change perspectives. We embrace the
commitment to reach net-zero carbon emissions in Canada. And also with
strong business acumen, our team members realized there would be a
growing demand on used vehicles for various reasons, such as collection,
preference for fossil fuel vehicles, etc. We should get ready to have a
robust business plan.

To initialize the business journey, we must learn how to evaluate used
cars and know which models would be more popular (in other words, will
generate better returns on our investment). As current MDS students, we
all understand how valuable a data-driven decision making approach is.
So the team searched from public accessible data sources and found the
1990s Car Evaluation dataset. Based on the dataset, we have trained and
optimized a classification machine learning algorithm to predict how
popular a given car model is. This would be much likely to boost our
efficiencies when starting to assess a large number of used cars on a
daily basis.

## Methods

### Data

The 1990s Car Evaluation dataset \[2\] utilized in this project was
primarily collected in June 1997 by Marko Bohanec to evaluate HINT
(Hierarchy INduction Tool), which is presented in B. Zupan, M. Bohanec,
I. Bratko, J. Demsar: Machine learning by function decomposition.
ICML-97, Nashville, TN. 1997. The original data can be downloaded from
the
<a href="https://archive-beta.ics.uci.edu/dataset/19/car+evaluation" target="_blank">UC
Irvine Machine Learning Repository</a> and follows an
<a href="https://creativecommons.org/licenses/by/4.0/legalcode" target="_blank">Creative
Commons Attribution 4.0 International License</a>.

There are 1728 observations in the dataset. Each row represents an
individual car with one target and six input attributes. The target
includes four classes such as “unacc”, “acc”, “good”, and “vgood”. The
six input attributes are overall price (buying), price of the
maintenance (maint), number of doors (doors), capacity in terms of
persons to carry (persons), the size of luggage boot (lug_boot) and the
safety of the car (safety).

### Question of Interest

Our research question is **given attributes of car models from the
1990s, how can we predict the popularity of each car model?**

This is a straightforward prediction question to be answered based on
the representative sample training data, sophisticated machine learning
models, and of course some domain experiences from one of our team
members. We all agree the research is a critical stepping stone to
success in our future business.

### Analysis

The 1990s Car Evaluation dataset has provided us with the privilege to
evaluate used vehicle based on six pre-selected categorical attributes.
Our objective is to take necessary steps to build a well-performed
machine learning model that can give us the best prediction result from
the inputs of the six attributes. In order to achieve the objective, the
very first step was to read the data, and split the data into training
and test sets to ensure the Machine Learning Golden Rule to be followed.
Second, we have used some basic `Python` \[3\] functions from `Pandas`
package \[4\] to quickly exam the dataset to know the columns, data
types, and if there is any missing value. After getting a higher level
understanding of the dataset, a more detailed Exploratory Data Analysis
(EDA) has been performed via `Pandas` and `Altair`packages \[5\] to find
correlations and patterns between input attributes and target classes.
Then we moved on to the standard machine learning process to build up
our prediction model. The process included the necessary feature
pre-processing/transformation, pipeline building, classifier tuning,
hyperparameters tuning, and final performance assessment by applying
Scikit-learn machine learning packages \[6\] within our defined Python
environment.

The raw dataset and our processed data are available to be accessed from
<a href="https://github.com/UBC-MDS/Car-Acceptability-Prediction/tree/main/data" target="_blank">here</a>.
All the code used to perform the analysis and generate this report can
be found at
<a href="https://github.com/UBC-MDS/Car-Acceptability-Prediction/tree/main/" target="_blank">GitHub
Car Acceptability Prediction Repository</a>.

## Results & Discussion

In the EDA stage, we have created two plots to visually explore the
distribution of six input attributes together with the target column,
and also the counting combinations between input attributes and target
classes. From Figure 1 we can see all six input attributes are almost
identical distributed, while the classes in the target column are quite
imbalanced. Most of the observations are belonging to the class of
“unacc”. Only a few of them are labeled as “good” or “vgood”. This might
remind us we have to be careful for the quality of used cars when we
start to build our inventory.

<img src="../result/eda_plot2_dist.png" alt="Figure 1. Distribution of six input features and target" width="40%" style="display: block; margin: auto;" />

Figure 2 has been generated to show the counting combinations between
six input features and the target. This is similar to the correlation
plot that can give us ideas how the input feature related to different
classes of target. For example, from the counting combination between
the feature of “persons” and the target, we can get an intuition that
small cars that only accommodate two people will be more unacceptable.
The finding also indicates the feature of “persons” is a meaningful
input for models to classify cars into appropriate classes of target.

<img src="../result/eda_plot1_corr.png" alt="Figure 2. Counting combinations between classes with each feature" width="60%" style="display: block; margin: auto;" />

Since the raw data was clean enough, our data processing was just
focusing on the training/test sets splitting and automatically saving
the separate dataframe in the designed data subfolder. Then in the model
building stage, we have applied `OrdinalEncoder` transformer to
pre-process all six input features, used four different classifiers,
`DummyClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`,
and `MultinomialNB` to perform classifier tuning by using
`balanced_accuracy` as the scoring metric. The classifier tuning result
is shown in Table 1.

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<caption>
Table 1. Classifier tuning results
</caption>
<thead>
<tr>
<th style="text-align:left;">
fit_time
</th>
<th style="text-align:left;">
score_time
</th>
<th style="text-align:left;">
test_score
</th>
<th style="text-align:left;">
train_score
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
0.003 (+/- 0.001)
</td>
<td style="text-align:left;">
0.002 (+/- 0.001)
</td>
<td style="text-align:left;">
0.250 (+/- 0.000)
</td>
<td style="text-align:left;">
0.250 (+/- 0.000)
</td>
</tr>
<tr>
<td style="text-align:left;">
0.003 (+/- 0.000)
</td>
<td style="text-align:left;">
0.001 (+/- 0.000)
</td>
<td style="text-align:left;">
0.949 (+/- 0.013)
</td>
<td style="text-align:left;">
1.000 (+/- 0.000)
</td>
</tr>
<tr>
<td style="text-align:left;">
0.058 (+/- 0.000)
</td>
<td style="text-align:left;">
0.005 (+/- 0.000)
</td>
<td style="text-align:left;">
0.959 (+/- 0.023)
</td>
<td style="text-align:left;">
1.000 (+/- 0.000)
</td>
</tr>
<tr>
<td style="text-align:left;">
0.003 (+/- 0.000)
</td>
<td style="text-align:left;">
0.001 (+/- 0.000)
</td>
<td style="text-align:left;">
0.283 (+/- 0.023)
</td>
<td style="text-align:left;">
0.284 (+/- 0.009)
</td>
</tr>
</tbody>
</table>

From Table 1, we can see `RandomForestClassifier` has the highest test
score (on the third row) in this case. Therefore, we decided to select
`RandomForestClassifier` as our prediction model and conducted the
random search hyperparameter tuning on this classifier. The hyperparamer
tuning was focused on two hyperparameters, rf\_\_n_estimators and
rf\_\_max_depth.

Table 2 has listed the hyperparameter optimization result (only first 6
rows displayed). It was sorted by the average test scores. As we can see
from Table 2, the test score has been further improved from 0.959 to
0.968, with the best hyperparameters (param_rf\_\_n_estimators equals to
68 and param_rf\_\_max_depth equals to 12). Our selected classifier
performed well on training dataset.

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<caption>
Table 2. Hyperparameter tuning results
</caption>
<thead>
<tr>
<th style="text-align:right;">
mean_test_score
</th>
<th style="text-align:right;">
param_rf\_\_n_estimators
</th>
<th style="text-align:right;">
param_rf\_\_max_depth
</th>
<th style="text-align:right;">
mean_fit_time
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
0.9680314
</td>
<td style="text-align:right;">
68
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
0.0577662
</td>
</tr>
<tr>
<td style="text-align:right;">
0.9679980
</td>
<td style="text-align:right;">
60
</td>
<td style="text-align:right;">
13
</td>
<td style="text-align:right;">
0.0570519
</td>
</tr>
<tr>
<td style="text-align:right;">
0.9668125
</td>
<td style="text-align:right;">
61
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
0.0601817
</td>
</tr>
<tr>
<td style="text-align:right;">
0.9665821
</td>
<td style="text-align:right;">
59
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
0.0525115
</td>
</tr>
<tr>
<td style="text-align:right;">
0.9650398
</td>
<td style="text-align:right;">
54
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
0.0546312
</td>
</tr>
<tr>
<td style="text-align:right;">
0.9638313
</td>
<td style="text-align:right;">
67
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
0.0664804
</td>
</tr>
</tbody>
</table>

As the last step in the model building stage, we applied out best model
on the test dataset and received the test score with 0.965. It is a
rewarding result.

There are some **limitations** and **future improvements** in this
analysis that should be highlighted:

1.  The results are based on the input features pre-processing,
    transformations, model tuning, and hyperparameters optimizations
    that have been applied in the report. The authors acknowledge that
    there might be other types of estimators and/or feature engineering
    methods that have not been considered here but may perform a better
    prediction.

2.  Although the optimistic scores both from the training and test data
    have been received, considering the size of the dataset is
    relatively small (1728 observations) and it only contains samples
    collected in 1990s, we are planning to search more available dataset
    to let our classifier can generalize better on unseen data before we
    employ it for our business.

## References

<div id="refs" class="references csl-bib-body">

<div id="ref-2035_zero_emission" class="csl-entry">

<span class="csl-left-margin">1. </span><span
class="csl-right-inline">Zhongming Z, Linong L, Xiaona Y, et al (2021)
Government launches consultations on commitment to require all new cars
sold in canada be zero emission by 2035</span>

</div>

<div id="ref-Car_evaluation_dataset" class="csl-entry">

<span class="csl-left-margin">2. </span><span
class="csl-right-inline">(1997) Car Evaluation</span>

</div>

<div id="ref-Python" class="csl-entry">

<span class="csl-left-margin">3. </span><span
class="csl-right-inline">Van Rossum G, Drake FL (2009) Python 3
reference manual. CreateSpace, Scotts Valley, CA</span>

</div>

<div id="ref-pandas" class="csl-entry">

<span class="csl-left-margin">4. </span><span
class="csl-right-inline">team T pandas development (2020)
[Pandas-dev/pandas: pandas](https://doi.org/10.5281/zenodo.3509134).
Zenodo</span>

</div>

<div id="ref-Altair" class="csl-entry">

<span class="csl-left-margin">5. </span><span
class="csl-right-inline">VanderPlas J, Granger B, Heer J, et al (2018)
Altair: Interactive statistical visualizations for python. Journal of
open source software 3:1057</span>

</div>

<div id="ref-scikit-learn" class="csl-entry">

<span class="csl-left-margin">6. </span><span
class="csl-right-inline">Pedregosa F, Varoquaux G, Gramfort A, et al
(2011) Scikit-learn: Machine learning in Python. Journal of Machine
Learning Research 12:2825–2830</span>

</div>

</div>
