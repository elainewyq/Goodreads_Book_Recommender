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


def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
        df: a dataframe.
        holdout_fraction: fraction of the latest dataframe rows to use in the test set.
    Returns:
        train: dataframe for training
        test: dataframe for testing
    """
    sample = int(round(len(df)*holdout_fraction,0))
    test = df[-sample:]
    train = df[~df.index.isin(test.index)]
    return train, test


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
        print('_train_init')
        with self._loss.graph.as_default():
            self._opt = optimizer(learning_rate)
            self._train_op = self._opt.minimize(self._loss)
            local_init_op = tf.group(
            tf.compat.v1.variables_initializer(self._opt.variables()),
            tf.compat.v1.local_variables_initializer())
            self._session = tf.compat.v1.Session()
            with self._session.as_default():
                self._session.run(tf.compat.v1.global_variables_initializer())
                self._session.run(tf.compat.v1.tables_initializer())
                local_init_op.run()
                tf.train.start_queue_runners()
        print('_train_init done')
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
                    ax.set_title('dnn model performance over num of iterations')
                    ax.legend()
                plt.savefig('DNN_model_perf.png')
        return results 

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

def book_neighbors(cleaned_books, model, title_substring, measure=COSINE, k=6):
    """Search for book ids that match the given substring.
    Args:
    model: MFmodel object.
    title_substring: match the given substring
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
    cleaned_books: cleaned book meta data which only contains book that exist in book review dataset
    cleaned_reviews: cleaned review data
    Returns:
    a dataframe with k entries of most relevant books
    """
    tmp_df = cleaned_books[cleaned_books['title'].str.contains(title_substring)]
    if len(tmp_df) == 0:
        print("Found no books with title %s" % title_substring) #update to print() and recommend popular items
        return
    book_id = tmp_df.sort_values('ratings_count', ascending=False).iloc[0].book_id
    title = cleaned_books[cleaned_books.book_id == book_id].title    
    print("Nearest neighbors of : %s." % title)

 
    scores = compute_scores(
      model.embeddings["book_id"][book_id], model.embeddings["book_id"],
      measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
      score_key: scores,
      'book_id': cleaned_books['book_id'],
      'titles': cleaned_books['title'],
    'average_rating': cleaned_books['average_rating'],
    'ratings_count': cleaned_books['ratings_count'],
    'link': cleaned_books['link']
    })
    display.display(df.sort_values([score_key], ascending=False).head(k))
    return df

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