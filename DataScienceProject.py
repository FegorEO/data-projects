#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import locale
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, validation_curve, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score


# In[2]:


# Unemployment data : https://www.bls.gov/data/
unemp_rate = pd.read_csv('bls_table.csv')
unemp_rate.head(10)


# In[3]:


# Job openings and hires data : https://www.bls.gov/data/
job_open = pd.read_csv('bls_table (1).csv')
job_open.head(10)


# In[4]:


# Merging both dataframes
df = pd.merge(unemp_rate, job_open, how='left')
df.head()


# In[5]:


# Picking variables that I will use for the data frame
new_df = df[['Date', 'Total', 'Total nonfarm job openings', 'Total nonfarm hires', 'Total private job openings', 'Total private hires', 'Government job openings', 'Government hires']]
new_df.head()


# In[6]:


# Renaming columns
compl_df = new_df.rename(columns={"Total": "Total Unemployment Rate"})
compl_df.head()


# In[7]:


# Changing date format

date1 = "2015-01"  # input start date
date2 = "2022-10"  # input end date

month_list = [i.strftime("%m-%y") for i in pd.date_range(start=date1, end=date2, freq='MS')]
compl_df['month_list'] = month_list
compl_df.head()


# In[8]:


# Removing commas from the numbers and transforming variable type
df = compl_df.replace(',','', regex = True)
print(df.dtypes)


# In[9]:


# Removing commas from the numbers and transforming variable type
df = compl_df.replace(',','', regex = True)


# In[10]:


# Removing October 2022 from dataframe
df = df[:-1]
df.head()


# In[11]:


# Transforming variable types to integers
df['Total nonfarm job openings'] = df['Total nonfarm job openings'].astype(int)
df['Total nonfarm hires'] = df['Total nonfarm hires'].astype(int)
df['Total private job openings'] = df['Total private job openings'].astype(int)
df['Total private hires'] = df['Total private hires'].astype(int)
df['Government job openings'] = df['Government job openings'].astype(int)
df['Government hires'] = df['Government hires'].astype(int)


print(df.dtypes)


# In[12]:


# Moving columns

cols = list(df.columns.values) 
cols.pop(cols.index('Date')) 
cols.pop(cols.index('month_list')) 
df = df[['month_list','Date']+cols] 
df.head()


# In[13]:


# Dropping 'Date' column
dff = df.drop(['Date'], axis=1)


# In[14]:


# Renaming date column and using 'date/time' format
df = dff.rename(columns={"month_list": "Date"})
df.head()

min_month = "2015-01"
max_month = "2022-09"

Date = pd.period_range(min_month, max_month, freq='M')
print(Date)

df['Date'] = Date
df.head(5)


# In[15]:


# Adding new date column to dataframe
cols = list(df.columns.values) 
cols.pop(cols.index('Date')) 
df = df[['Date']+cols] #Create new dataframe with columns in the order you want
df.head()
print(df.dtypes)


# In[16]:


# Removing hyphen from date column
Date = df['Date'].astype(str).replace('-','', regex = True)
Date


# In[17]:


# Creating binary variable welfare_01 using a forloop and applying if/else condition
Welfare_01 = []
for values in df['Total Unemployment Rate']:
    if values < 5: # Healthy unemployment rate according to economists
        Welfare_01.append(0)
    else:
        Welfare_01.append(1)

# Adding new variable to column
Welfare_01        
df['Welfare_01'] = Welfare_01
df.head(5)


# In[18]:


# Replace date for data visualization
Unemployment = df['Total Unemployment Rate']


# In[19]:


# Finding the average unemployment rate of each year
year_15 = df['Total Unemployment Rate'].iloc[0:11].mean()
year_16 = df['Total Unemployment Rate'].iloc[12:23].mean()
year_17 = df['Total Unemployment Rate'].iloc[24:35].mean()
year_18 = df['Total Unemployment Rate'].iloc[36:47].mean()
year_19 = df['Total Unemployment Rate'].iloc[48:59].mean()
year_20 = df['Total Unemployment Rate'].iloc[60:71].mean()
year_21 = df['Total Unemployment Rate'].iloc[72:83].mean()
year_22 = df['Total Unemployment Rate'].iloc[84:].mean()
year_15


# In[20]:


df.describe(include = 'all')


# In[21]:


variables = df[['Total nonfarm job openings','Total nonfarm hires','Total private job openings','Total private hires','Government job openings','Government hires']]
variables.head()


# In[22]:


# For visualizing the unemployment rate for each year
df_2 = [year_15, year_16, year_17, year_18, year_19, year_20, year_21, year_22]
df_2 = pd.DataFrame({'Avg_Unemployment_Rate':df_2})
years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
df_2['years'] = years
df_2


# In[148]:


# Line plot of unemployment rate from 2015-2022
ax = plt.subplot()
ax.plot(df_2['years'], df_2['Avg_Unemployment_Rate'])
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
plt.title('Unemployment Rate from 2015 to Sep 2022', size=14, loc='left')
plt.xlabel('Years')
plt.ylabel('Total Unemployment Rate')

plt.savefig('Unemployment_Rate_by_year.png')


# In[149]:


# Scatter plot of nonfarm jop openings and unemployment rate
x = df['Total nonfarm job openings']
y = df['Total Unemployment Rate']
plt.scatter(x,y)
plt.xlabel('Total nonfarm job openings')
plt.xticks(rotation = 67)
plt.ylabel('Total Unemployment Rate',)
plt.title('Unemployment Rate against Nonfarm job openings from 2015 to 2022')


plt.savefig('Unemployment_v_jobopenings.png')


# In[23]:


# Target array and feature matrix
y = df['Welfare_01'] 
X = df[['Total Unemployment Rate','Total nonfarm job openings', 'Total nonfarm hires', 'Total private job openings', 'Government hires', 'Government job openings']]


# In[24]:


# Splitting data into training and test size
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=10)


# In[25]:


# Fitting for logistic regression
logit = LogisticRegression()
logit.fit(Xtrain, ytrain)


# In[26]:


# Creating predictions
logit_predictions = logit.predict(Xtest)
logit_prob = logit.predict_proba(Xtest)
print(logit_predictions)
print(logit_prob[::,1])


# In[27]:


# Conducting a confusion matrix and checking accuracy score of test data
print('Accuracy:', logit.score(Xtest, ytest))
confusion_matrix(ytest, logit.predict(Xtest))


# In[28]:


# Conducting a confusion matrix and checking accuracy score of train data
logit.fit(Xtrain, ytrain)
print('Logit Training score: ', logit.score(Xtest, ytest))
y_pred = logit.predict_proba(Xtest)[::,1]
confusion_matrix(ytrain, logit.predict(Xtrain))


# In[29]:


# Checking the AUC 
false_positive_rate, true_positive_rate, thresholds = roc_curve(ytest, y_pred)
logit_rates = pd.DataFrame(dict(fpr=false_positive_rate, tpr=true_positive_rate))
logit_auc = auc(logit_rates['fpr'], logit_rates['tpr'])
print('Logit AUC: ', logit_auc) #AUC


# In[30]:


# Graphing the ROC
fig3 = plt.figure() # this allows for layering
plt.plot(logit_rates.fpr, logit_rates.tpr, 'b', label = 'Logit')
plt.plot([0, 1], [0, 1],'r--')  # 'r--' for red and dashed
plt.xlim([0, 1]) # sets limit for the axis
plt.ylim([0, 1]) 
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right') # legend into the lower right hand corner
plt.savefig('logit.png')


# In[46]:


#Scaling the data for KNN
scaler = StandardScaler()
Xsc = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
Xsc.head()


# In[49]:


#Splitting into training and test data
Xsctrain, Xsctest, ytrain, ytest = train_test_split(Xsc, y, test_size=0.2, random_state=10)


# In[50]:


#Fitting for KNN
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(Xsctrain, ytrain)
cross_val_score(knn, Xsc, y, cv=5)


# In[65]:


knn_predictions = knn.predict(Xsctest)
#knn_prob = knn.predict_proba(Xtest)
print(knn_predictions)
#print(knn_prob[::,1])


# In[59]:


#Checking accuracy score for test data
k_range = np.arange(1,11)
train_scores, test_scores = validation_curve(knn, Xsc, y, param_name='n_neighbors',
                                            param_range=k_range, cv=10)

print('Accuracy:', knn.score(Xsctest, ytest))


# In[162]:


#Checking accuracy score for training data
k_range = np.arange(1,11)
train_scores, test_scores = validation_curve(knn, Xsc, y, param_name='n_neighbors',
                                            param_range=k_range, cv=10)

print('Accuracy:', knn.score(Xsctrain, ytrain))


# In[163]:


#Generalization scores for KNN
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_mean


# In[164]:


# Validation curve for KNN
plt.plot(k_range, train_mean, label='Train')
plt.plot(k_range, test_mean, label='Test')
plt.xlabel('k')
plt.ylabel('Score')
plt.title('KNN Validation Curve')
plt.legend()

plt.savefig('knn.png')

