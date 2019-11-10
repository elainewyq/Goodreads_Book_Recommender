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
                self._session.run(tf.tables_initializer())
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
                    ax.set_title('collaborative filtering model performance over num of iterations')
                    ax.legend()
                plt.savefig('collaborative_filtering_model_perf.png')
        return results 
