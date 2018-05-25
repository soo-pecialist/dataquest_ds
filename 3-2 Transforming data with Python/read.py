import pandas as pd

def load_data():
	result = pd.read_csv("hn_stories.csv", 
		names = ["submission_time", "upvotes", "url", "headline"])
	return result
	
	#data = load_data()
	#print(data.head())
	


	

