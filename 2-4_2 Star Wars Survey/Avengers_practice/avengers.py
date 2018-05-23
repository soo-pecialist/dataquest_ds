#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:09:44 2018

@author: KimSooHyeon
"""
import pandas as pd
import numpy as nd
import matplotlib.pyplot as plt

avengers = pd.read_csv("avengers.csv", encoding='latin-1')
true_avengers = pd.DataFrame()
## Show histogram
avengers['Year'].hist();
plt.show()

## We only want to keep the Avengers who were introduced after 1960
true_avengers = avengers[avengers['Year'] >= 1960].copy()

## Combine "Death1,2,3,4,5" 
def clean_deaths(row):
    num_deaths = 0
    columns = ["Death1", "Death2", "Death3", "Death4", "Death5"]
    
    for col in columns:
        death = row[col]
        if pd.isnull(death) or death == 'NO':
            continue
        elif death == 'YES':
            num_deaths += 1
    return num_deaths 

true_avengers["Deaths"] = true_avengers.apply(clean_deaths, axis=1)

## we want to verify that the Years since joining field accurately reflects the Year column
joined_accuracy_count  = int()

def years_calculator(row):
    reference_year = 2015
    how_long = row["Years since joining"]
    year = row["Year"]
    if reference_year - year == how_long:
        return 1
    else:
        return 0
joined_accuracy_count = true_avengers.apply(years_calculator, axis=1).sum()
""" another way:
correct_joined_years = true_avengers[true_avengers['Years since joining'] == (2015 - true_avengers['Year'])]
joined_accuracy_count = len(correct_joined_years)
"""
print(joined_accuracy_count)