import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score



dataset = pd.read_csv('/content/drive/MyDrive/Financial-Data.csv')

dataset.head()
dataset.columns
dataset.describe()

# Removing NaN
dataset.isna().any()
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=14)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()#gca stands for get current axes
    f.set_title(dataset2.columns.values[i])
#the vals variable holds all the uniqe values of the column and
#also helps tp sacle the graph. lets say u are ploting for age
#100 then vals not only identifies the unique age with100 values
#but also scales the graphs according to it
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
#the condition is to make sure that it doesnt take much time
#to plot the graphs at the same time ensuring that ur system does not crash
#depending upon how powerfull ur system is. So if the unique values in vals exceed 100
#the bins would be defaulted to 100.
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#the layout is for rectangular show play with it to better fit ur screen
#also if you want make a few histograms at a time
## Correlation with Response Variable (Note: Models like RF
#are not linear like these)
#corrwith creates a corelation within the columns

#bar creats the bargraph
dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize = (20, 4), title = "Correlation with E Signed", fontsize = 8,
        rot = 30, grid = True, color='red' )
#just play with the numbers to better fit the graph in your screen
#the negative and positive side of the graph shows the relation between the variable and the target
#negative means inverse and positive means direct relation
#in this the target is e-sign

## Correlation Matrix
#set -Set the aesthetic style of the plots. the parameters are style and rc basically make the backgroud
#style is the kind of plot we will get it can changed to dark,ticks,whitegrid etc also its kind of a dictionary
#rc- Parameter mappings to override the values in the preset seaborn style dictionaries.
#This only updates parameters that are considered part of the style definition.
sn.set(style="white")

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
#Return an array of zeros with the same shape and type as a given array.
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)
#cmap is short for colormap
# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

