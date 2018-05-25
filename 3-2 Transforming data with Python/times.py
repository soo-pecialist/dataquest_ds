import read
import pandas as pd
from dateutil.parser import parse
import datetime

df = read.load_data()

def dates(x):
	y = parse(x)
	return y.hour

df['hour'] = df['submission_time'].apply(dates)
print(df['hour'].value_counts().head())	
