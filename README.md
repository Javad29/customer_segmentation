# Aim
The aim of this learning project is to find out whether customers can be clustered into different groups, based on their transactions. Furthermore it would be interesting to know, the defining characteristics of each cluster.

# Data set
The data set can be found/downloaded here: https://archive.ics.uci.edu/ml/datasets/Online+Retail
The data set contains all the transactions for a UK-based and registered non-store online retail that occured between December 01/12/2010 and 09/12/2010. The products that are sold by the company are mainly unique all-occassion gifts.

# Outcomes
Customers of cluster 1 seem to be customers with a relatively recent purchasing history, those customers also seem to buy relatively frequently and also to spend more. 

Customers belonging to clusters 0 and 2 are customers whose purchasing history lies somewhere in the middle. The distinguishing factor between the two clusters is frequency. Customers belonging to cluster 2 purchase less frequently, customers belonging to cluster 0 however have a higher purchasing frequency. Customers from both clusters seem to spend equally.

The purchases of customers from cluster 3 seem to be relatively dated and the customers in this cluster do spend relatively little irrespective of their purchasing frequency. 

# Further ideas
Generally speaking there seem to be "low-revenue" customers (cluster 3) and "high-revenue" customers (cluster 1). It would be interesting to try figuring out how to nudge custemers from cluster 3 into behaving more like customers from cluster 1 i.e. to buy more frequently and with shorter purchasing intervals.

# Sources
- Sarkar, D., Bali, R., Sharma, T. (2018). Practical Machine learning with Python: A problem-solver's guide to building real-world intelligent systems. Apress
-   https://www.kaggle.com/code/naren3256/kmeans-clustering-and-cluster-visualization-in-3d/notebook