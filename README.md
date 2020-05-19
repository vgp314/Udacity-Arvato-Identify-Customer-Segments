
# Udacity Arvato Identify Customer Segments

# Capstone Project


## Table of Contents
1. Problem Statement and Project Motivation
2. Instalation
3. Files
4. Results and Conclusions
5. Licenses and Acknowledgements
6. Project Motivation

## Problem Statement
- Consider a company's marketing campaign (Arvato Financial Services), in which we
need to select those individuals who can become the company's future customers. For this task,
we have the following databases: demographic information from Germany (country where the
company is located) and information from individuals who are already customers of this
company.<br />
- First, the demographic information of the German population was analyzed in order to
understand and explore the main characteristics of this population.<br />
- Then, we create a predictive model that can determine with reasonable accuracy whether
a person can become a possible consumer of the company, when subjected to a certain
marketing campaign.<br />
- Finally, we classify each possible consumer, from an unexplored test database, and
submit the result on the kaggle platform.<br />

## Project Motivation
- The project is a problem for a company, with real data and with several possible approaches. It is a rich set of data and an interesting problem to be solved. Submitting work on Kaggle is a way to compare the quality of our algorithm with that of other students. So I chose to do this specific job that motivated me to learn even more.




## Instalation

- The following packages are necessary: numpy , datetime, pandas , matplotlib, seaborn , math, sklearn , pylab ,itertools, imblearn, pickle, xgboost

## Files
- project.pdf - Report with detailed explanation of the entire project.<br />
- capstone_proposal.pdf - Report with a proposal for thus project.<br />
- util.py - python module with basically data processing and feature engineering <br />
- cluster.py -- python module with clustering methods for segmentation report <br />
- pca.py -- python module with pca methods for dimensionality reduction <br />
- Udacity_AZDIAS_052018.csv: Demographics data for the general population of
Germany; 891 211 persons (rows) x 366 features (columns);
- Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order
company; 191 652 persons (rows) x 369 features (columns);
- Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were
targets of a marketing campaign; 42 982 persons (rows) x 367 (columns);
- Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were
targets of a marketing campaign; 42 833 persons (rows) x 366 (columns);
- unknown_values.csv: Mapping dictionary with attributes and the value of the unkown value

## Results and Conclusions

- The result of this work can be found in the file final_project.pdf, as well as any details of implementation, conclusions and future work

## Licenses and Acknowledgements

- The project is part of Udacity's machine learning nanodegree program. The data provided is not public, and belongs to Arvato and Udacity



