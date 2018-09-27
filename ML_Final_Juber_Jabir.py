# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:28:12 2018

@author: Md Juber Rahman(U00617285) and Md Jabir Rahman(U00655625)

"""
# Import all the required libraries/packages
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

# Print a part of the dataset to help picking a Customer ID
print("A portion of Data set is printed to help in picking Customer ID, Movie ID, Valid Year of release etc. but you may pick any value from the dataset")
print("\n")
print(df_rating.iloc[::30000,:])
print("\n")
print(df_title.iloc[::3000,:])

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
print("\n-:Common Input:-")
customerID = input("Enter User(Customer) ID Please: ")
customerID = int(customerID)
print("\nInput for part 1 - Predicted Score for a Movie given a User ID:")
print("User ID given by you",customerID)

enteredMovieId=input("Enter Movie ID Please: ")
enteredMovieId=int(enteredMovieId)

print("\nInput for part 2 - Recommendation of top scored movie released in a given year for a given User ID:")
print("User ID given by you",customerID)
releaseYear = input("Enter the year of movie release: ")
releaseYear = int(float(releaseYear))
print("Please wait for the output...")
number = 92

sdata = df[(df['Cust_Id']== customerID)]
sdata = sdata.set_index('Movie_Id')
sdata = sdata.join(df_title)['Name']

user = df_title.copy()
user = user.reset_index()
user = user[~user['Movie_Id'].isin(drop_movie_list)]
data = Dataset.load_from_df(df[['Movie_Id', 'Cust_Id', 'Rating']], reader)
trainset = data.build_full_trainset()
algo.fit(trainset)


user['Estimated_Score'] = user['Movie_Id'].apply(lambda x: algo.predict(customerID, x).est)
user = user.sort_values('Estimated_Score', ascending = False)
print("\n\nResults for part 1: Predicted Score-")
print("Recommending for the given user the predicted rating of the Movie ID: ", enteredMovieId)
predictedScore=user.loc[user['Movie_Id'] ==int(enteredMovieId) ]
print(predictedScore.to_string(index=False))
N=1
pd.set_option('display.width', 1000)
endList=(user.loc[user['Year'] ==int(releaseYear) ])

# part 2: recommend movie for the user
print("\n\nResults for part 2: Recommendation-")
print("Recommending for the given user the top scored movie released in the year: ", releaseYear)
topRated=endList.head(N)
print(topRated.to_string(index=False))
