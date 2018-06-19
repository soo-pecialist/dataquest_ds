#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:56:48 2018

@author: Soo Hyeon Kim

Description:
    this project will focus on credit modelling, a well known data science 
    problem that focuses on modeling a borrower's credit risk. Credit has 
    played a key role in the economy for centuries and some form of credit has 
    existed since the beginning of commerce. We'll be working with financial 
    lending data from Lending Club. Lending Club is a marketplace for personal 
    loans that matches borrowers who are seeking a loan with investors looking 
    to lend money and make a return. 

"""

#### Itroduction
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import os.path

loans_2007 = pd.read_csv('LoanStats3a.csv', skiprows=1)
loans_2007 = loans_2007.drop(['desc', 'url'], axis=1)
half_count = len(loans_2007) / 2
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)

fname = 'loans_2007.csv'
file_present = os.path.isfile(fname)

if not file_present:
    loans_2007.to_csv('loans_2007.csv', index=False)
else:
    print("file exists\n")

### <Data Cleaning>
## cleaning columns - remove data leak, redundant columns
drop_cols = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 
            'sub_grade', 'emp_title', 'issue_d', 'debt_settlement_flag']

for col in drop_cols:
    if col in loans_2007.columns.values:
        loans_2007.drop(columns = col, inplace=True)

drop_cols = ['zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 
             'total_pymnt_inv', 'total_rec_prncp']

loans_2007.drop(columns=drop_cols, inplace=True)

drop_cols = ['total_rec_int', 'total_rec_late_fee', 'recoveries', 
             'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt']

loans_2007.drop(columns=drop_cols, inplace=True)

#print(loans_2007.columns.values)

## check our target
#loans_2007['loan_status'].value_counts()

loans_2007 = loans_2007[(loans_2007['loan_status'] == "Fully Paid") | \
                        (loans_2007['loan_status'] == "Charged Off")]

status_replace = {
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0,
    }
}

loans_2007 = loans_2007.replace(status_replace)

## drop columns that contains only one unique value
drop_cols = list()

for col in loans_2007.columns:
    series = loans_2007[col].dropna() # nan does not count
    if len(series.unique()) == 1:
        drop_cols.append(col)

#print(drop_columns)
loans_2007.drop(columns=drop_cols, inplace=True)

### <Preparing the Featrues>
## handling missing values
loans = loans_2007.copy() # filtered df

#null_counts = loans.isnull().sum()
# print(null_counts[null_counts>0])

loans.drop(['pub_rec_bankruptcies'], axis=1, inplace=True) # bc > 1%
loans.dropna(axis=0, inplace=True)
#print(loans.dtypes.value_counts())

object_columns_df = loans.select_dtypes(include=['float'])

# Let's explore the unique value counts of the columns that seem like containing categorical values

#cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']
#for c in cols:
#    print(loans[c].value_counts())

#print(loans["purpose"].value_counts())
#print(loans["title"].value_counts())
"""
    It seems like the purpose and title columns do contain overlapping information 
    but we'll keep the purpose column since it contains a few discrete values. 
    In addition, the title column has data quality issues since many of the values 
    are repeated with slight modifications 
    (e.g. Debt Consolidation and Debt Consolidation Loan and debt consolidation)
"""

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loans.drop(['last_credit_pull_d', 'addr_state', 'title', 'earliest_cr_line'], axis=1, inplace=True)
cols = ['int_rate', 'revol_util']
for col in cols:
    loans[col] = loans[col].str.rstrip('%').astype(float)
loans = loans.replace(mapping_dict)
"""
    Let's now encode the home_ownership, verification_status, purpose, 
    and term columns as dummy variables so we can use them in our model
"""
cols = ['home_ownership', 'verification_status', 'term', 'purpose']

dummy_df = pd.get_dummies(loans[cols])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cols, axis=1)

#### Making Prediction
"""
    we should optimize for:
        - high recall (true positive rate)
        - low fall-out (false positive rate)
"""

target = loans['loan_status']
features = loans[loans.columns.drop('loan_status')]

penalty = {
    0: 10,
    1: 1
}

lr = LogisticRegression(class_weight=penalty)

#lr = LogisticRegression(class_weight='balanced')
predictions = cross_val_predict(lr, features, target)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)