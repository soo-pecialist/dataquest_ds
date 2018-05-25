import read
import pandas as pd

df = read.load_data()
domains = df['url'].value_counts()
top100 = domains[:100]

for name, counts in top100.items():
	print("{0}: {1}".format(name, counts))

