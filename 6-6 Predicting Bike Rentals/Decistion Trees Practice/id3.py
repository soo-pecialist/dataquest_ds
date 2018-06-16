import numpy as np
import pandas as pd
import math

###### codes ########################
def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    See reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    # Compute the counts of each unique value in the column
    counts = np.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)
    
    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    
    return -entropy



def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a data set, column to split on, and target
    See reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])
    
    # Find the median of the column we're splitting
    column = data[split_name]
    median = column.median()
    
    # Make two subsets of the data, based on the median
    left_split = data[column <= median]
    right_split = data[column > median]
    
    # Loop through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * calc_entropy(subset[target_name])
    
    # Return information gain
    return original_entropy - to_subtract


def find_best_column(data, target_name, columns):
    # Automatically find the column in columns to split on
    # i.e., which column returns maximum information gain
    # data is a dataframe
    # target_name is the name of the target variable
    # columns is a list of potential columns to split on
    
    information_gains = list()
    for col in columns:
        information_gains.append(calc_information_gain(data, col, target_name))
    
    best = columns[information_gains.index(max(information_gains))]
       
    return best

########### Decisiotn Tree (ID3 algorithm)

### Pseudocode 
""" ID3 algorithm (reference: https://en.wikipedia.org/wiki/ID3_algorithm)
def id3(data, target, columns)
    1 Create a node for the tree
    2 If all values of the target attribute are 1, Return the node, add 1 to counter_1
    3 If all values of the target attribute are 0, Return the node, add 1 to counter_0
    4 Using information gain, find A, the column that splits the data best
    5 Find the median value in column A
    6 Split column A into values below or equal to the median (0), and values above the median (1)
    7 For each possible value (0 or 1), vi, of A,
    8    Add a new tree branch below Root that corresponds to rows of data where A = vi
    9    Let Examples(vi) be the subset of examples that have the value vi for A
   10    Below this new branch add the subtree id3(data[A==vi], target, columns)
   11 Return Root
"""

# Create a dictionary to hold the tree  
# It has to be outside of the function so we can access it later
tree = {}

# This list will let us number the nodes  
# It has to be a list so we can access it inside the function
nodes = []

def id3(data, target, columns, tree):
    unique_targets = pd.unique(data[target])
    
    # Assign the number key to the node dictionary
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        if unique_targets == 1:
            tree["label"] = 1
        else:
            tree["label"] = 0

        return
    
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()
    
    # Insert code here that assigns the "column" and "median" fields to the node dictionary  
    tree["column"] = best_column
    tree["median"] = column_median
    
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]
    
    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])

#### print function
"""
	e.g., print_node(tree, 0)

	age > 37.5
    age > 25.0
        age > 22.5
            Leaf: Label 0
            Leaf: Label 1
        Leaf: Label 1
    age > 55.0
        age > 47.5
            Leaf: Label 0
            Leaf: Label 1
        Leaf: Label 0
"""

def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))
    
    
def print_node(tree, depth):
    # Check for the presence of "label" in the tree
    if "label" in tree:
        # If found, then this is a leaf, so print it and return
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        # This is critical -- without it, you'll get infinite recursion
        return
    # Print information about what the node is splitting on
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    
    # Create a list of tree branches
    branches = [tree["left"], tree["right"]]
    print_node(branches[0], depth+1)
    print_node(branches[1], depth+1)
    # Insert code here to recursively call print_node on each branch
    # Don't forget to increment depth when you pass it in


def predict(tree, row):
    if "label" in tree:
        return tree["label"]
    
    column = tree["column"]
    median = tree["median"]
    
    if row[column] <= median:
    # If it's less than or equal, return the result of predicting on the left branch of the tree
        return predict(tree["left"], row)
    else:
    # If it's greater, return the result of predicting on the right branch of the tree
        return predict(tree["right"], row)

def batch_predict(tree, df):
    """
		predictions = batch_predict(tree, dataframe)
		print(predictions)
    """
    predictions = df.apply(lambda row: predict(tree, row), axis=1)
    
    return predictions


"""
Examples of tree (dict)
{  
   "left":{  
      "left":{  
         "left":{  
            "number":4,
            "label":0
         },
         "column":"age",
         "median":22.5,
         "number":3,
         "right":{  
            "number":5,
            "label":1
         }
      },
      "column":"age",
      "median":25.0,
      "number":2,
      "right":{  
         "number":6,
         "label":1
      }
   },
   "column":"age",
   "median":37.5,
   "number":1,
   "right":{  
      "left":{  
         "left":{  
            "number":9,
            "label":0
         },
         "column":"age",
         "median":47.5,
         "number":8,
         "right":{  
            "number":10,
            "label":1
         }
      },
      "column":"age",
      "median":55.0,
      "number":7,
      "right":{  
         "number":11,
         "label":0
      }
   }
}
"""
################



## [Test]
## Create the data set that we used in the example on the last screen
data = pd.DataFrame([
    [0,20,0],
    [0,60,2],
    [0,40,1],
    [1,25,1],
    [1,35,2],
    [1,55,1]
    ])
# Assign column names to the data
data.columns = ["high_income", "age", "marital_status"]
id3(data, "high_income", ["age", "marital_status"], tree)

print(nodes); print()
print(tree); print()

print_node(tree, 0); print()

## prediction
new_data = pd.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
    ])
# Assign column names to the data
new_data.columns = ["age", "marital_status"]

predictions = batch_predict(tree, new_data)
print(predictions)



