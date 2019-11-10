import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
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

def count_reviews(file_name):
    """
    read the file name and return how many reviews, books and users
    in the review dataset
    """
    print('counting file:', file_name)
    n_review = 0
    book_set, user_set = set(), set()
    print('current line: ', end='')
    with gzip.open(file_name) as f:
        for l in f:
            d = json.loads(l)
            if n_review % 1000000 == 0:
                print(n_review, end=',')
            n_review += 1
            book_set.add(d['book_id'])
            user_set.add(d['user_id'])
    print('complete')
    print('done!')
    return n_review, len(book_set), len(user_set)


def read_reviews(file_name):

    """
    read the reviews dataset which is in json.gz format and convert to pandas
    dataframe

    """

    print('counting file:', file_name)
    n_review = 0
    book_ids = []
    user_ids = []
    review_ids = []
    ratings = []
    #review_texts = []
    timestamp = []
    n_votes = []
    n_comments = []
    
    print('current line: ', end='')
    with gzip.open(file_name) as f:
        for l in f:
            d = json.loads(l)
            if n_review % 1000000 == 0:
                print(n_review, end=',')
            n_review += 1
            review_ids.append(d['review_id'])
            book_ids.append(d['book_id'])
            user_ids.append(d['user_id'])
            ratings.append(d['rating'])
            # review_texts.append(d['review_text'])
            timestamp.append(d['date_added'])
            n_votes.append(d['n_votes'])
            n_comments.append(d['n_comments'])            
    print('complete')
    print('done!')
    return pd.DataFrame({'review_id': review_ids, 'user_id': user_ids, 'book_id': book_ids, 'rating':ratings,
                        #'review_text': review_texts, 
                        'timestamp':timestamp, 'n_votes': n_votes, 'n_comments':n_comments})

def read_books(file_name, head = False, num_books = 500000):
    print('counting file:', file_name)
    book_ids = []
    work_ids = []
    isbn = []
    asin = []
    titles = []
    description = []
    num_pages = []
    is_ebook = []
    links = []
    country_code = []
    language_code = []
    average_rating = []
    ratings_count = []
    text_reviews_count = []
    author_id = []
    publisher = []
    publication_year = []
    genre = []
    similar_books = []
    
    n_books = 0
    print('current line: ', end='')
    with gzip.open(file_name) as f:
        for l in f:
            d = json.loads(l)
            if n_books % 500000 == 0:
                print(n_books, end=',')
            n_books += 1
            
            book_ids.append(d['book_id'])
            work_ids.append(d['work_id'])
            isbn.append(d['isbn'])
            asin.append(d['asin'])
            titles.append(d['title'])
            description.append(d['description'])
            num_pages.append(d['num_pages'])
            is_ebook.append(d['is_ebook'])
            links.append(d['link'])
            country_code.append(d['country_code'])
            language_code.append(d['language_code'])
            average_rating.append(d['average_rating'])
            ratings_count.append(d['ratings_count'])
            text_reviews_count.append(d['text_reviews_count'])
            author_id.append(d['authors'])
            publisher.append(d['publisher'])
            publication_year.append(d['publication_year'])
            
            #top user-generated shelves for a book, used to define genres for this book by Goodreads
            genre.append(d['popular_shelves'])
            similar_books.append(d['similar_books'])  #a list of book ids that users who like the current book also like
            
            #option to read the first num_books of books
            if head:                                  
                if n_books > num_books:
                    break
           
    print('complete')
    print('done!')
    return pd.DataFrame({'book_id': book_ids, 'work_id': work_ids, 'isbn': isbn, 'asin':asin,
                        'title': titles, 'description': description, 'num_pages':num_pages, 'is_ebook': is_ebook,
                        'link': links, 'country_code': country_code, 'language_code': language_code,
                        'average_rating': average_rating, 'ratings_count': ratings_count,
                        'text_reviews_count': text_reviews_count, 'author_id': author_id,
                        'publisher': publisher, 'publication_year': publication_year,'genre': genre, 'similar_books': similar_books})
                


def clean_reviews(reviews_df):
    """
    clean review data
    """

    reviews_df1 = reviews_df[reviews_df.rating != 0]
    reviews_df1.timestamp = reviews_df1.timestamp.apply(lambda x: convert_timestamp(x))
    reviews_df2 = reviews_df1.sort_values('timestamp')
    start = pd.to_datetime('2000-01-01 00:00:00')
    df = reviews_df2[reviews_df2.timestamp>=start].copy()

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
        # genre_df.reset_index(inplace=True)
    books_df.drop(['genre'], axis =1, inplace=True)
        # books_df.reset_index(inplace=True)
        # df = pd.concat([books_df, genre_df],axis=1)
    

    #author_id is a dictionary with keys 'author_id' and 'role' and we only keep the first author
    books_df.author_id = books_df.author_id.apply(lambda x:x[0]['author_id'])


    books_df.language_code = books_df.language_code.replace(r'^\s*$', np.nan, regex=True)
    books_df1 = books_df[books_df.language_code.apply(lambda x:True if x in ['eng', 'en-GB', 'en-CA', 'en-US', 'en', np.nan] else False)].copy()
    
    books_df1.drop(['work_id','isbn','asin', 'country_code', 'language_code', 'publication_year', 'description'], axis =1, inplace=True)  

    return books_df1

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


def review_map_id(df):
    # remap the user id and book ids so they start from 0.
    df['old_user_id'] = df['user_id']
    df['user_id'] = map_id(df['old_user_id'])

    df['old_book_id'] = df['book_id']
    df['book_id'] = map_id(df['old_book_id'])

    return df

def book_map_id(df):
    # remap the book ids so they start from 0.
    df['old_book_id'] = df['book_id']
    df['book_id'] = map_id(df['old_book_id'])
    return df

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
            if int(i['count']) >= 5:
                genre += i['name'] +' '
        return genre
    except:
        return 'mystery thriller crime'

def convert_timestamp(x):
    x_new = x[4:19] + x[-5:]
    return datetime.strptime(x_new, '%b %d %H:%M:%S %Y')


def read_tables(data_dir, reviews_filename, books_filename):

    #load review data
    reviews_df = read_reviews(os.path.join(data_dir, reviews_filename))
    cleaned_reviews = clean_reviews(reviews_df)
    #delete users that only have one rating in review dataset
    mask = cleaned_reviews.groupby('user_id').count()['rating'] > 1
    keep_user_ids= cleaned_reviews.groupby('user_id').count()['rating'][mask].index
    cleaned_reviews1 = cleaned_reviews[cleaned_reviews.user_id.isin(keep_user_ids)]
    #delete books that only have one rating in review dataset
    mask_book = cleaned_reviews1.groupby('book_id').count()['rating'] > 1
    keep_book_ids= cleaned_reviews1.groupby('book_id').count()['rating'][mask_book].index
    cleaned_reviews2 = cleaned_reviews1[cleaned_reviews1.book_id.isin(keep_book_ids)]
    book_in_reviews = cleaned_reviews2.book_id.unique()

    #load book meta data
    books_df = read_books(os.path.join(data_dir, books_filename), head=False)
    cleaned_books = clean_books(books_df)
    book_ids = cleaned_books.book_id.unique() #we deleted all non-english books in book meta data when clean book data

    #keep books that exist in both cleaned reviews and cleaned books
    total_book = np.intersect1d(book_ids, book_in_reviews)
    cleaned_books1 = cleaned_books[cleaned_books.book_id.isin(total_book)]
    cleaned_reviews3 = cleaned_reviews2[cleaned_reviews2.book_id.isin(total_book)]

    # remap the book_id and user_id in review table for matrix factorization
    # remap the book_id in book table to make it consistent with review table

    return review_map_id(cleaned_reviews3), book_map_id(cleaned_books1)