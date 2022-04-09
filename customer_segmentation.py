# Python version 3 or higher
# -*- coding: utf-8 -*- 
"""
@author: Javad Mousavi Nia (Javad29)
"""

import os
import math
import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tabulate import tabulate

## Checking whether data set exists:

if os.path.exists('datensatz online retail.xlsx'):
    print('Data set exists')
else:
    print('Data set does not exist and can be downloaded from: https://archive.ics.uci.edu/ml/datasets/Online+Retail')

## Loading the data set:

data_df=pd.read_excel(io=r'datensatz online retail.xlsx')  
                                                         
## Data analysis:

print('What follows is a brief analysis of the data set')

top_10_destinations=data_df.Country.value_counts().head(n=10)

print(f"Top 10 destinations for the company (by country):{os.linesep}{os.linesep.join(map(str,top_10_destinations.index))}")

print(f"The number of transactions for the top destination (UK) was: {top_10_destinations.get(key='United Kingdom')}")

print(f"The number of transactions for the second destination (Germany) was: {top_10_destinations.get(key='Germany')}")

print(f"The company has {data_df.CustomerID.unique().shape[0]} customers")

top_10_customers=dict((data_df.CustomerID.value_counts()/sum(data_df.CustomerID.value_counts())*100).head(n=10))

top_10_customers={int(k):round(v,2) for k,v in top_10_customers.items()}

print(f"These are the ID's of the top 10 customers and the relative size of their transactions: {os.linesep}{top_10_customers}")

print(f"There are {data_df.CustomerID.isnull().sum()} missing values for customer ID")

quantity_statistics=dict(data_df.Quantity.describe())

print(f"The minimum value in terms of quantity is negative: {quantity_statistics['min']}")

unit_price=data_df.UnitPrice.describe()

print(f"The minimum value in terms of unit_price is negative: {unit_price['min']}")

print('The negative values may have ben caused by product returns') 

## Data preparation and data cleansing

print('As the UK is the biggest market for the company all further analysis will be concentrated on that market')

data_gb_df=data_df[data_df.Country=="United Kingdom"]

print(data_gb_df.head())

data_gb_df["Revenue"]=data_df.UnitPrice*data_df.Quantity

data_gb_df_first_five=data_gb_df.head()

print(f'A new column "Revenue" has been created defined as unit price * quantity (excerpt):' 
f'{tabulate(data_gb_df_first_five.loc[:,["Quantity","UnitPrice","Revenue"]], headers="keys", tablefmt="psql")}')

data_gb_df=data_gb_df[~(data_gb_df.Revenue<0)]

returns=data_gb_df.apply(lambda x: True if x['Revenue']<0 else False, axis=1)

num_rows=len(returns[returns==True].index)

if num_rows >= 1:
    print(f'There are still {num_rows} entries in the data set with a negative revenue')
else:
    print('Entries with negative revenue have been successfully removed')

data_gb_df=data_gb_df[~(data_gb_df.CustomerID.isnull())]

no_customerid = data_gb_df['CustomerID'].isna().sum()

print(f'Number of missing customer ids after removal: {no_customerid}')

## Feature creation:

reference_date=data_gb_df.InvoiceDate.max()
reference_date=reference_date+datetime.timedelta(days=1)

print(f'A reference date has been created defined as the day which is after the day of the last transaction: {reference_date}')

data_gb_df['days_since_last_purchase']=reference_date-data_gb_df.InvoiceDate

print(f'A new variable (column) which represents the number of days prior to the reference date was created:' 
f'{tabulate(data_gb_df.loc[:,["days_since_last_purchase"]].head(), headers="keys", tablefmt="psql")}')

data_gb_df['Recency']=data_gb_df['days_since_last_purchase'].astype('timedelta64[D]')

print(f'The column "days_since_last_purchase" has been changed into a numerical column Recency'
f'{tabulate(data_gb_df.loc[:,["Recency"]].head(), headers="keys", tablefmt="psql")}')

customer_history_df=data_gb_df.groupby("CustomerID").min().reset_index()[["CustomerID", "Recency"]]

print(f'A new data frame customer_history_df was created:'
f'{tabulate(customer_history_df.head(), headers="keys", tablefmt="psql")}')

customer_monetray_val=data_gb_df[['CustomerID', 'Revenue']].groupby('CustomerID').sum().reset_index()

customer_history_df=customer_history_df.merge(customer_monetray_val, how='outer')
customer_history_df.Revenue=customer_history_df.Revenue+0.001

print(f"The monetary value of a customer (sum of the customer's genereated revenue) has been joined with customer_history_df:"
f'{tabulate(customer_history_df.head(), headers="keys", tablefmt="psql")}')

customer_freq=data_gb_df[['CustomerID', 'Revenue']].groupby('CustomerID').count().reset_index()

customer_freq.rename(columns={'Revenue':'Frequency'}, inplace=True)

customer_history_df=customer_history_df.merge(customer_freq, how='outer')

print(f"A new feature Frequency (count of customer's transactions) has been joined with customer_history_df:"
f'{tabulate(customer_history_df.head(), headers="keys", tablefmt="psql")}')

## Data Preprocessing:

customer_history_df['recency_log']=customer_history_df['Recency'].apply(math.log)

customer_history_df['frequency_log']=customer_history_df['Frequency'].apply(math.log)

customer_history_df['revenue_log']=customer_history_df['Revenue'].apply(math.log)

feature_vector=['recency_log','frequency_log', 'revenue_log']

print(f"In order to reduce the range of values the logarithms of the features of customer history were taken:"
f'{tabulate(customer_history_df.head(), headers="keys", tablefmt="psql")}')

x=customer_history_df[feature_vector].values
scaler=preprocessing.StandardScaler().fit(x)
x_scaled=scaler.transform(x)

print(f'The logarithmic features of customer history were standardized for the proper functioning of the k-means algorithm:'
f'{tabulate(x_scaled[:3,:], tablefmt="psql")}')

## Clustering:

n=7

int_list=[]
for i in range(1,n):
    model=KMeans(n_clusters=i, init='k-means++')
    model.fit(x_scaled)
    int_list.append(model.inertia_)
fig=plt.figure(figsize=(8,6))
plt.plot(range(1,n),int_list, linewidth=3, marker='o',color = 'dodgerblue')
plt.xlabel("Number of Clusters")
plt.ylabel("Minimal Squared Residuals in Cluster")
print('Among the range of 7 (experimentally chosen) clusters, 4 seems to be the ideal number of clusters:')
plt.show()

model = KMeans(n_clusters = 4, init = "k-means++", max_iter = 300, n_init = 10, random_state = 2018)
cluster = model.fit_predict(x)

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[cluster == 0,0],x[cluster == 0,1],x[cluster == 0,2], s = 20 , color = 'blue', label = "cluster 0")
ax.scatter(x[cluster == 1,0],x[cluster == 1,1],x[cluster == 1,2], s = 20 , color = 'orange', label = "cluster 1")
ax.scatter(x[cluster == 2,0],x[cluster == 2,1],x[cluster == 2,2], s = 20 , color = 'green', label = "cluster 2")
ax.scatter(x[cluster == 3,0],x[cluster == 3,1],x[cluster == 3,2], s = 20 , color = 'red', label = "cluster 3")

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Revenue')
ax.legend()

print('Under an ideal cluster number of 4 the data set would be clustered in the following way:')
plt.show()
     


