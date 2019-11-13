# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:51:07 2019

@author: Dheeraj Kumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

r_column = ['user_id', 'movie_id', 'rating', 'timestamp']
u_column = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
m_column = ['movie_id', 'title_year', 'genre' ]


movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', names = m_column, encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', names = u_column, encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', names = r_column, encoding = 'latin-1')

movies.info()
users.info()
ratings.info()

ratings['rating'].min() 
ratings['rating'].max() 

movies.isnull().any()
ratings.isnull().any()
users.isnull().any()

drama_movies=movies['genre'].str.contains('Drama')
movies[drama_movies].head()
drama_movies.shape

movie_data_ratings_data=movies.merge(ratings,on = 'movie_id',how = 'inner')
movie_data_ratings_data.head(10)

high_rated= movie_data_ratings_data['rating']>3.0
movie_data_ratings_data[high_rated].head(10)

movies['year'] =movies['title_year'].str.extract('.*\((.*)\).*',expand = False)
movies.head(3)

def count_word(df, ref_col, listt):
    keyword_count = dict()
    for s in listt: keyword_count[s] = 0
    for listt_keywords in df[ref_col].str.split('|'):
        if type(listt_keywords) == float and pd.isnull(listt_keywords): continue
        for s in listt_keywords: 
            if pd.notnull(s): keyword_count[s] += 1
   
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

genre_labels = set()
for s in movies['genre'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))
        
keyword_occurences, sam = count_word(movies, 'genre', genre_labels)
keyword_occurences
trunc_occurences = keyword_occurences[0:50]

fig = plt.figure(1, figsize=(18,13))
ax2 = fig.add_subplot(2,1,2)
y_axis = [i[1] for i in trunc_occurences]
x_axis = [k for k,i in enumerate(trunc_occurences)]
x_label = [i[0] for i in trunc_occurences]
plt.xticks(rotation=75, fontsize = 12)
plt.yticks(fontsize = 12)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of records", fontsize = 24, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center', color='b')
plt.title("Popularity of Genres",fontsize = 30)
plt.show()


training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')
