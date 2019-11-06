import gzip
import json
import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

import tensorflow as tf
import altair as alt
import collections

# sparse representation of the rating matrix
def build_rating_sparse_tensor(rating_df, users_num, books_num):
    indices = rating_df[['user_id', 'book_id']].values
    values = rating_df['rating'].values
    return tf.SparseTensor(indices=indices, values=values, dense_shape=[users_num, books_num])



#sparse means square error
# only gather the embeddings of the observed pairs and compute the errors 
# rather than compute the full prediction matrix to save computation power

def sparse_mean_square_error(sparse_ratings, user_embeddings, book_embeddings):

    """
    Args:
        sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
        user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
        dimension, such that U_i is the embedding of user i.
        book_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
        dimension, such that V_j is the embedding of book j.
    Returns:
        A scalar Tensor representing the MSE between the true ratings and the
        model's predictions.
    """
    predictions = tf.reduce_sum(
      tf.gather(user_embeddings, sparse_ratings.indices[:, 0]) *
      tf.gather(book_embeddings, sparse_ratings.indices[:, 1]),
      axis=1)
    loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
    return loss

def gravity(U, V):
  """Creates a gravity loss given two embedding matrices."""
  return 1. / (U.shape[0].value*V.shape[0].value) * tf.reduce_sum(
      tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))

class CFModel(object):
    """Simple class that represents a collaborative filtering model"""
    """
    Initializes a CFModel.
    Args:
    embedding_vars: A dictionary of tf.Variables.
    loss: A float Tensor. The loss to optimize.
    metrics: optional list of dictionaries of Tensors. The metrics in each
    dictionary will be plotted in a separate figure during training.
    """
    def __init__(self, embedding_vars, loss, metrics=None,
                learning_rate=1.0, 
                optimizer=tf.compat.v1.train.AdamOptimizer):
        self._embedding_vars = embedding_vars
        self._loss = loss
        self._metrics = metrics
        self._embeddings = {k: None for k in embedding_vars}
        self._train_init(learning_rate, optimizer)

    @property
    def embeddings(self):
        """The embeddings dictionary."""
        return self._embeddings

    def _train_init(self, learning_rate, optimizer):
        with self._loss.graph.as_default():
            self._opt = optimizer(learning_rate)
            self._train_op = self._opt.minimize(self._loss)
            local_init_op = tf.group(
            tf.variables_initializer(self._opt.variables()),
            tf.local_variables_initializer())
            self._session = tf.Session()
            with self._session.as_default():
                self._session.run(tf.global_variables_initializer())
                self._session.run(tf.tables_initializer())
                local_init_op.run()
                tf.train.start_queue_runners()

    def train(self, num_iterations=100, plot_results=True):
        """
        Trains the model.
        Args:
        iterations: number of iterations to run.
        learning_rate: optimizer learning rate.
        plot_results: whether to plot the results at the end of training.
        optimizer: the optimizer to use. Default to GradientDescentOptimizer.
        Returns: The metrics dictionary evaluated at the last iteration.
        """
        
        with self._loss.graph.as_default(), self._session.as_default():
            
            iterations = []
            metrics = self._metrics or ({},)
            metrics_vals = [collections.defaultdict(list) for _ in self._metrics]
            
            # Train and append results.
            for i in range(num_iterations + 1):
                _, results = self._session.run((self._train_op, metrics))
                if (i % 1 == 0) or i == num_iterations:
                    print("\r iteration %d: " % i + ", ".join(
                    ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                    end='')
                    iterations.append(i)
                    for metric_val, result in zip(metrics_vals, results):
                        for k, v in result.items():
                            metric_val[k].append(v)
            
            for k, v in self._embedding_vars.items():
                self._embeddings[k] = v.eval()

            if plot_results:
                # Plot the metrics.
                num_subplots = len(metrics)+1
                fig = plt.figure()
                fig.set_size_inches(num_subplots*10, 8)
                for i, metric_vals in enumerate(metrics_vals):
                    ax = fig.add_subplot(1, num_subplots, i+1)
                    for k, v in metric_vals.items():
                        ax.plot(iterations, v, label=k)
                    ax.set_xlim([1, num_iterations])
                    ax.set_xlabel('num_iterations')
                    ax.set_title('train/test error over num of iterations')
                    ax.legend()
        return results

# Utility to split the data into training and test sets.
def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
        df: a dataframe.
        holdout_fraction: fraction of the latest dataframe rows to use in the test set.
    Returns:
        train: dataframe for training
        test: dataframe for testing
    """
    # test = df.sample(frac=holdout_fraction, replace=False) 
    # train = df[~df.index.isin(test.index)]
    # return train, test
    # df.sort_values('timestamp', inplace=True)
    sample = int(round(len(df)*holdout_fraction,0))
    test = df[-sample:]
    train = df[~df.index.isin(test.index)]
    return train, test

def build_model(ratings, embedding_dim=3, init_stddev=1., regularization_coeff=0, gravity_coeff=0, optimizer=tf.compat.v1.train.AdamOptimizer):
    """
    Args:
        ratings: a DataFrame of the ratings
        embedding_dim: the dimension of the embedding vectors.
        init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
        model: a CFModel.
    """
    users_num = len(ratings['user_id'].unique())
    books_num = len(ratings['book_id'].unique())
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings, users_num, books_num)
    A_test = build_rating_sparse_tensor(test_ratings, users_num, books_num)
    # Initialize the embeddings using a normal distribution.
    U = tf.Variable(tf.random_normal(
      [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
      [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
    train_loss = sparse_mean_square_error(A_train, U, V)
    test_loss = sparse_mean_square_error(A_test, U, V)
    regularization_loss = regularization_coeff * (
        tf.reduce_sum(U*U)/U.shape[0].value + tf.reduce_sum(V*V)/V.shape[0].value)
    gravity_loss = gravity_coeff * gravity(U, V)    
    total_loss = train_loss + regularization_loss + gravity_loss
    metrics = {
      'train_error': train_loss,
      'regularization_error': regularization_loss,
      'gravity_loss': gravity_loss,
      'test_error': test_loss
    }
    embeddings = {
      "user_id": U,
      "book_id": V
    }

    return CFModel(embeddings, total_loss, [metrics], optimizer)


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
    ids =  cleaned_books[cleaned_books['title'].str.contains(title_substring)].index.values
    print(ids)
    titles = cleaned_books.loc[ids]['title'].values
    if len(titles) == 0:
        raise ValueError("Found no books with title %s" % title_substring)
    print("Nearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
        print("[Found more than one matching book. Other candidates: {}]".format(
        ", ".join(titles[1:])))
    book_id = cleaned_reviews[cleaned_reviews.old_book_id == ids[0]].book_id
    scores = compute_scores(
      model.embeddings["book_id"][book_id], model.embeddings["book_id"],
      measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
      score_key: scores[0],
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
    'ratings_count': cleaned_books['ratings_count']
    })
    display.display(df.sort_values([score_key], ascending=False).head(k))