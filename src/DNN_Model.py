import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
from collections import defaultdict
import tensorflow as tf
import altair as alt
import collections

from .utils import split_dataframe, CFModel
from .load_data import read_tables

def generate_dataset(reviews, books, min_good_rating =4, max_num_books_per_user=20):

    """generate dataset for dnn model"""
    MIN_GOOD_RATING = min_good_rating
    MAX_NUM_BOOKS_PER_USER = max_num_books_per_user
    
    prev_good_books = defaultdict(set) # from user_id to a set of book_ids
    prev_bad_books = defaultdict(set)
    
    reviews_books = reviews.merge(books, how='left', on=['book_id'])
    
    reviews_books['book_id'] = reviews_books['book_id'].astype(str)
    reviews_books['user_id'] = reviews_books['user_id'].astype(str)
    reviews_books['author_id'] = reviews_books['author_id'].astype(str)
    
    
    data = []
    reviews_books = reviews_books.sort_values(by=['timestamp'])
    for i in range(reviews_books.shape[0]):
        if i % 1000 == 0:
            print("\r processed %d rows " % i, end='')
        row = reviews_books.iloc[i]
        user_id = row['user_id']
        book_id = row['book_id']
        rating = row['rating']
        
        entry = {
            'user_id': user_id,
            'book_id': book_id,
            'rating': rating,
            'timestamp': row['timestamp'],
            'author_id': row['author_id'],
            'prev_good_books':  list(prev_good_books[user_id]), # make a copy for each example
            'prev_bad_books': list(prev_bad_books[user_id])
        }
        # TODO: remove this check
        # if len(prev_good_books[user_id]) > 0 and len(prev_bad_books[user_id]) > 0:
        data.append(entry)
        
        if rating >= MIN_GOOD_RATING and len(prev_good_books[user_id])< MAX_NUM_BOOKS_PER_USER:
            prev_good_books[user_id].add(book_id)
        elif rating < MIN_GOOD_RATING and len(prev_bad_books[user_id]) < MAX_NUM_BOOKS_PER_USER:
            prev_bad_books[user_id].add(book_id)
    return pd.DataFrame(data)


def make_batch(ratings, batch_size):
    """Creates a batch of examples.
    Args:
        ratings: A DataFrame of ratings such that examples["book_id"] is a list of
        books rated by a user.
    batch_size: The batch size.
    """
    def pad(x, fill):
        return pd.DataFrame.from_dict(x).fillna(fill).values

    features = {
      "prev_good_books": pad(ratings['prev_good_books'].values.tolist(), ""),
      "prev_bad_books": pad(ratings['prev_bad_books'].values.tolist(), ""),        
      "rating": ratings['rating'].values,
       "book_id": ratings['book_id'].values,
        "author_id": pad(ratings['author_id'].values.tolist(), "")
      }
    print('make_batch#3')    
    batch = (
      tf.data.Dataset.from_tensor_slices(features)
      .shuffle(1000)
      .repeat()
      .batch(batch_size)
      .make_one_shot_iterator()
      .get_next())
    print('make_batch#4')    
    return batch

def make_shared_embedding_col(keys, shared_name, vocabulary_list, embedding_dim):
    """create shared embedding col for previously liked and disliked book_ids"""
    columns = []
    for key in keys:
        columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
            key, vocabulary_list, num_oov_buckets=0))

    return tf.feature_column.shared_embedding_columns(
        columns, shared_embedding_collection_name=shared_name, dimension=embedding_dim)

def make_embedding_col(key, vocabulary_list, embedding_dim):
    """create embedding col for book content features"""
    categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=key, vocabulary_list=vocabulary_list, num_oov_buckets=0)
    return tf.feature_column.embedding_column(
      categorical_column=categorical_col, dimension=embedding_dim,
      # default initializer: trancated normal with stddev=1/sqrt(dimension)
      combiner='mean')
    
def mean_square_error_loss(batch_user_embeddings, book_embeddings, batch_book_ids, labels):
    """loss function for dnn model"""
    user_emb_dim = batch_user_embeddings.shape[1].value
    book_emb_dim = book_embeddings.shape[1].value
    if user_emb_dim != book_emb_dim:
        raise ValueError('The user embedding dimension %d should match the book embedding dimension %d' %(
                user_emb_dim, book_emb_dim))
        
    batch_book_embeddings = tf.gather(book_embeddings, tf.strings.to_number(batch_book_ids, out_type=tf.dtypes.int32))
    
    batch_predictions = tf.reduce_sum(
        batch_user_embeddings * batch_book_embeddings, axis=1)
    
    loss = tf.compat.v1.losses.mean_squared_error(labels, batch_predictions)
    return loss

def build_dnn_model(dataset, embedding_cols, hidden_dims, learning_rate =1,):
    print('build_dnn_model#1 create_network')
    def create_network(features):
        #create a bog-of-words embedding for each sparse feature
        inputs = tf.compat.v1.feature_column.input_layer(features, embedding_cols)
        #hidden layer
        input_dim = inputs.shape[1].value
        for i, output_dim in enumerate(hidden_dims):
            w = tf.get_variable(
                'hidden%d_w_'% i, shape=[input_dim, output_dim],
                initializer=tf.truncated_normal_initializer(
                stddev=1./np.sqrt(output_dim)))/10
            outputs = tf.matmul(inputs, w)
            input_dim = output_dim
            inputs = outputs
        return outputs
    
    print('build_dnn_model#2 split train_test dataset')  
    train_dataset, test_dataset = split_dataframe(dataset)
    print('train_data mean rating: %.5f'%train_dataset.rating.mean())

    train_batch = make_batch(train_dataset, 200)
    test_batch = make_batch(test_dataset, 100)

    print('build_dnn_model#3')    
    with tf.compat.v1.variable_scope('model', reuse=False):
        #train
        train_user_embeddings =create_network(train_batch)
        train_labels = train_batch['rating']
        train_book_ids = train_batch['book_id']
        
    with tf.compat.v1.variable_scope('model', reuse=True):
        #test
        test_user_embeddings = create_network(test_batch)
        test_labels = test_batch['rating']
        test_book_ids = test_batch['book_id']
        
        book_embeddings = tf.get_variable("input_layer/book_id_embedding/embedding_weights")
        
    print('build_dnn_model#4 calculate train/test loss')        
    train_loss = mean_square_error_loss(train_user_embeddings, book_embeddings, train_book_ids, train_labels)
    test_loss = mean_square_error_loss(test_user_embeddings, book_embeddings, test_book_ids, test_labels)
    
#     _, test_prediction_at_10 = tf.metrics.precision_at_k(
#         labels=test_labels, predictions=tf.matmul(test_user_embeddings, book_embeddings, transpose_b=True),
#         k=10)
    
    metrics=(
        {'train_loss': train_loss, 'test_loss': test_loss},
        # {'test_precision_at_10': test_prediction_at_10}
    )
    embeddings = {'book_id': book_embeddings}
    return CFModel(embeddings, train_loss, metrics, learning_rate=learning_rate)


DOT = 'dot'
COSINE = 'cosine'
def compute_scores(query_embedding, item_embeddings, measure=COSINE):
    """Computes the scores of the candidates given a query.
    Args:
        query_embedding: a vector of shape [k], representing the query embedding.
        item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
        measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
        scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    if measure == COSINE:
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    return scores

def book_neighbors(cleaned_books, cleaned_reviews, model, title_substring, measure=COSINE, k=6):
    """Search for book ids that match the given substring.
    Args:
    model: MFmodel object.
    title_substring: a string that appears in a book name
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
    cleaned_books: cleaned book meta data which only contains book that exist in book review dataset
    cleaned_reviews: cleaned review data
    Returns:
    a dataframe with k entries of most relevant books
    """
    ids =  cleaned_books[cleaned_books['title'].str.contains(title_substring)].book_id.values
    titles = cleaned_books[cleaned_books.book_id.isin(ids)]['title'].values
    if len(titles) == 0:
        raise ValueError("Found no books with title %s" % title_substring) #update to print() and recommend popular items
    print("Nearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
        print("[Found more than one matching book. Other candidates: {}]".format(
        ", ".join(titles[1:])))
    book_id = ids[0]
    scores = compute_scores(
      model.embeddings["book_id"][book_id], model.embeddings["book_id"],
      measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
      score_key: scores,
      'titles': cleaned_books['title'],
    'is_ebook': cleaned_books['is_ebook'],
    'average_rating': cleaned_books['average_rating'],
    'ratings_count': cleaned_books['ratings_count']
    })
    display.display(df.sort_values([score_key], ascending=False).head(k))

def user_recommendations(cleaned_books, cleaned_reviews, model, user_id, measure=COSINE, k=6):
    """Search for book ids that have the highest predicted scores for the given user
    Args:
    model: MFmodel object.
    user_id: string - the original user_id
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
    cleaned_books: cleaned book meta data which only contains book that exist in book review dataset
    cleaned_reviews: cleaned review data
    Returns:
    a dataframe with k entries of the highly recommended books
    """
    ids =  cleaned_reviews[cleaned_reviews['old_user_id'] == user_id].user_id.values
    if len(ids) == 0:
        raise ValueError("Found no users with id %s" % user_id)
    print("The highest recommendations for user %s." % user_id)
    scores = compute_scores(
      model.embeddings["user_id"][ids[0]], model.embeddings["book_id"],
      measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
      score_key: scores,
      'titles': cleaned_books['title'],
    'is_ebook': cleaned_books['is_ebook'],
    'average_rating': cleaned_books['average_rating'],
    'ratings_count': cleaned_books['ratings_count'],
    'text_reviews_count': cleaned_books['text_reviews_count']
    })
    # display.display(df.sort_values([score_key], ascending=False).head(k))
    return df
