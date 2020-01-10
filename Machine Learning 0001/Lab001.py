'''
Created on 9 Jan. 2020

@author: NerminKuc
'''
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
dataset_raw = df.head()
print(dataset_raw)

# summarize the data
dataset_summary = df.describe()
print(dataset_summary)

#Lets select some features to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#we can plot each of these features:
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Now, lets plot each of these features vs the Emission, to see how linear is their relation:
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# do the same plot but this time engine-size vs emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#thats the way the cookie crumbles
plt.show()

# do the same plot but this time engine-size vs emissions
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='red')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
#thats the way the cookie crumbles
plt.show()

# https://docs.scipy.org/doc/numpy-1.15.1/user/basics.creation.html
# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.rand.html
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk] #80% - TRUE
test = cdf[~msk] #20% - FALSE


