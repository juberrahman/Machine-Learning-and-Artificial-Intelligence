#!bin/bash/

import pandas as pd
import numpy as np
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import cross_validate
from time import time
sns.set_style("darkgrid")

# Load data from the "ratings" file
df_rating = pd.read_csv('ratings.txt', header = None, names = ['Movie_Id', 'Cust_Id', 'Rating'], usecols = [0,1,2])
df_rating['Rating'] = df_rating['Rating'].astype(float)

# Load data from the file "movie title"
df_title = pd.read_csv('movie_titles.txt', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)

# Let's visualize some examples from the data set
df = df_rating
df.index = np.arange(0, len(df))

# Count the number of unique movies, unique customers and unique ratings
print("Summary of the Data set")
movie_count = df['Movie_Id'].nunique()
print("No. of unique Movies: ", movie_count)
customer_count = df['Cust_Id'].nunique()
print("No. of unique Customers ID: ", customer_count)
rating_count = df['Rating'].nunique()
print("No. of Rating Options: ", rating_count)

# Check for Missing Values
print("No. of missing values:", df.isnull().sum().sum())

# Print a part of a dataset to help picking a Customer ID
print("A portion of Data set is printed to help pick a Customer ID")
print(df_rating.iloc[::5000,:])

# Get some info about movie review and customer review
J = ['count', 'mean']
df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(J)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary)['count'].quantile(0.8)
drop_movie_list = df_movie_summary[df_movie_summary['count']<movie_benchmark].index

df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(J)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary)['count'].quantile(0.8)
drop_movie_list = df_cust_summary[df_cust_summary['count']<cust_benchmark].index

# pivot the data set to make a matrix form
df_pivot = pd.pivot_table(df, values = 'Rating', index = 'Cust_Id', columns = 'Movie_Id')

# Now start collaborative filtering
reader = Reader()

algo = SVD()

# Take input for Customer ID
customerID = input("Customer ID Please: ")
customerID = int(customerID)
print("We are recommending Movies for Customer ID: ", customerID)

number = input("Please input the total number of movies to be recommended: ")
number = int(number)
print("Number of Movies Being Recommended: ", number)

sdata = df[(df['Cust_Id']== customerID)]
print(sdata)
sdata = sdata.set_index('Movie_Id')
sdata = sdata.join(df_title)['Name']

user = df_title.copy()
user = user.reset_index()
user = user[~user['Movie_Id'].isin(drop_movie_list)]

data = Dataset.load_from_df(df[['Movie_Id', 'Cust_Id', 'Rating']], reader)
trainset = data.build_full_trainset()
algo.fit(trainset)

user['Estimated_Score'] = user['Movie_Id'].apply(lambda x: algo.predict(customerID, x).est)
user = user.drop('Movie_Id', axis = 1)
user = user.sort_values('Estimated_Score', ascending = False)
print(user.head(number))

