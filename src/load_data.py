import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
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