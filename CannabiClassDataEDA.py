#!/usr/bin/env python
# coding: utf-8

# # Getting Data and EDA

# ## Imported necessary libraries and loaded in df

# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import pickle
import warnings
warnings.filterwarnings('ignore')

fulldf=pickle.load( open( "Pickles/df.pickle", "rb" ) )


#Visualizing Indica, Sativa, and Hybrid. Describes what a Hybrid Strain Is.


plt.figure(figsize=(10,5))
venn2(subsets=(len(fulldf[fulldf['type']==0]),len(fulldf[fulldf['type']==1]),len(fulldf[fulldf['type']==2])),set_labels=('Indica','Sativa'))
plt.savefig('Visualizations/VennDiagram_Indica_Sativa.png')


effects=fulldf.columns[2:]


#Checking How THC Content Varies with Type


plt.figure(figsize=(10,5))
sns.catplot(x='type',y='thc',kind='violin',data=fulldf)
plt.savefig('Visualizations/ViolinPlots.png')


#Examining different kinds of effects v. thc content


plt.figure(figsize=(8,6))
plt.title('Positive Effects')
sns.scatterplot(x='thc',y='positive',hue='type',data=fulldf)
plt.savefig('Visualizations/positive.png')
plt.figure(figsize=(8,6))
plt.title('Negative Effects')
sns.scatterplot(x='thc',y='negative',hue='type',data=fulldf)
plt.savefig('Visualizations/negative.png')
plt.figure(figsize=(8,6))
plt.title('Medical Effects')
sns.scatterplot(x='thc',y='medical',hue='type',data=fulldf)
plt.savefig('Visualizations/medical.png')


#Checking for Class Imbalance



indicadf=fulldf[fulldf['type']==0]
sativadf=fulldf[fulldf['type']==1]
hybriddf=fulldf[fulldf['type']==2]

sns.distplot(fulldf['type'])
prind=len(indicadf)/len(fulldf)
prsat=len(sativadf)/len(fulldf)
prhyb=len(hybriddf)/len(fulldf)
print('Probability of Indica: {}'.format(prind))
print('Probability of Sativa: {}'.format(prsat))
print('Probability of Hybrid: {}'.format(prhyb))


#Looking for Important Features: Feature Selection By Hand

for e in effects:
    plt.figure(figsize=(10,5))
    sns.boxplot(x='type',y=e,data=fulldf)
    plt.savefig('Visualizations/{}.png'.format(e).replace(' ',''))
# sns.scatterplot(sativadf['Relaxed'])
# sns.scatterplot(hybriddf['Relaxed'])



for i in range(0, 38, 5):
    g = sns.PairGrid(fulldf,
                     x_vars = effects[i:i+5],
                     y_vars = ['type'])
    g = g.map(sns.kdeplot)



