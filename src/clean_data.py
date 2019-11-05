import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler



pd.options.display.float_format = '{:,}'.format
pd.set_option('display.max_colwidth', -1)


def clean_reviews(reviews_df):
    """
    clean review data
    """

    reviews_df1 = reviews_df[reviews_df.rating != 0]
    reviews_df1.timestamp = reviews_df1.timestamp.apply(lambda x: convert_timestamp(x))
    reviews_df2 = reviews_df1.sort_values('timestamp')
    start = pd.to_datetime('2000-01-01 00:00:00')
    df = reviews_df2[reviews_df2.timestamp>=start].copy()

    # remap the user id and book ids so they start from 0.
    print('remapping the ids')
    df['old_user_id'] = df['user_id']
    df['user_id'] = map_id(df['old_user_id'])

    df['old_book_id'] = df['book_id']
    df['book_id'] = map_id(df['old_book_id'])
    return df

def clean_books(books_df):
    """
    to clean books graph raw data
    input: book raw data
    """    
    #filter columns and clean columns we will keep
    books_df.num_pages = books_df.num_pages.replace(r'^\s*$', 0, regex=True)
    books_df.num_pages =books_df.num_pages.astype('int')
    books_df.average_rating =books_df.average_rating.astype('float')
    books_df.ratings_count =books_df.ratings_count.astype('int')
    books_df.text_reviews_count =books_df.text_reviews_count.astype('int')
    books_df.is_ebook = books_df.is_ebook.apply(lambda x: 1 if x=='true' else 0)

    #genre is a dictionary with keys 'count' and 'name'. we only keep 'name' with more than 2 counts.
        # books_df.genre = books_df.genre.apply(lambda x: genre_list(x))
        # count = CountVectorizer()
        # count_matrix = count.fit_transform(books_df.genre)
        # genre_df = pd.DataFrame(count_matrix.todense(), index=books_df.book_id)
    
    books_df.language_code

    books_df.drop(['work_id','isbn','asin', 'country_code', 'similar_books', 'link'], axis =1, inplace=True)
    
    ## revist the features below after creating a pipeline
    books_df.drop(['description', 'author_id', 'language_code', 'publisher', 'publication_year'], axis =1, inplace=True)

    #author_id is a dictionary with keys 'author_id' and 'role' and we only keep the first author
    # books_df.author_id = books_df.author_id.apply(lambda x:x[0]['author_id'])
     
    #one encoder for 'is_ebook' and 'author_id'
    #there are missing values in features 'languague_code' 'publisher' and 'publication_year' which are filled with 'empty str'. 
    #we will keep them as they are and encode

    #get enbedding for 'title', 'description' and 'genre'
        # genre_df.reset_index(inplace=True)
    books_df.drop(['genre'], axis =1, inplace=True)
        # books_df.reset_index(inplace=True)
        # df = pd.concat([books_df, genre_df],axis=1)
    books_df.set_index(['book_id'], drop=True, inplace=True)

    return books_df

def scale_data(df,label=False):
    sc = StandardScaler()
    indices = df.index
    cat_cols=[]
    cols = ['num_pages','average_rating','ratings_count','text_reviews_count']
    for i in df.columns:
        if i not in cols:
            cat_cols.append(i)
    scaled_df = pd.DataFrame(sc.fit_transform(df[cols]),columns=cols)
    scaled_df.reset_index(inplace=True)
    cat_df = df[cat_cols]
    cat_df.reset_index(inplace=True)
    final_df = pd.concat([scaled_df,cat_df],axis=1)
    final_df.drop(['index', 'book_id', 'title'],axis=1,inplace=True)
    final_df.index = indices
    return final_df


def map_id(s):
    unique_values = sorted(s.unique())
    mapping = {}
    for i in range(len(unique_values)):
        mapping[unique_values[i]] = i
    
    new_id = s.apply(lambda x: mapping[x])
    return new_id


def genre_list(x):
    genre = ''
    try:
        for i in x:
            if int(i['count']) >= 1:
                genre += i['name'] +' '
        return genre
    except:
        return 'mystery thriller crime'

def convert_timestamp(x):
    x_new = x[4:19] + x[-5:]
    return datetime.strptime(x_new, '%b %d %H:%M:%S %Y')
