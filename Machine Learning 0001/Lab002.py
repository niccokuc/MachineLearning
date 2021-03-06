'''
Created on 9 Jan. 2020

@author: NerminKuc
'''
'''
Created on 9 Jan. 2020

@author: NerminKuc
'''

#Train/Test Split involves splitting the dataset into training and testing sets respectively,
#which are mutually exclusive. After which, you train with the training set and test with the testing set.
#This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not
#part of the dataset that have been used to train the data. It is more realistic for real world problems

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Read in the data file
df = pd.read_csv("FuelConsumption.csv")
print(df)

# take a look at the dataset
dataset_raw = df.head()
#print(dataset_raw)

# summarize the data
dataset_summary = df.describe()
#print(dataset_summary)

# Lets select some features to explore more. Only select these features to use from the data.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#print(cdf)


# https://docs.scipy.org/doc/numpy-1.15.1/user/basics.creation.html
# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.rand.html
print(len(df))

# Creates an array of 1 dimension TRUE AND FALSE of the index of the data of the rows based on
# random numbers between 0 and 1067 for 80% of sample
msk = np.random.rand(len(df)) < 0.8 
print(msk)
# Using the array 'msk', overlay it against the data 'cdf' for 80% - TRUE
train = cdf[msk]
# Using the array 'msk', overlay it against the data 'cdf' for 20% - FALSE
test = cdf[~msk]
print(train)
print(test)

