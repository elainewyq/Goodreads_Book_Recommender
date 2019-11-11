import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

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
    df.sort_values('timestamp', inplace=True)
    sample = int(round(len(df)*holdout_fraction,0))
    test = df[-sample:]
    train = df[~df.index.isin(test.index)]
    return train, test


def get_error(rating_df):
    """Calculate root mean squared error.
    """
    sqerror = abs(rating_df['rating'] - rating_df['prediction']) ** 2  # squared error array
    mse_error = sqerror.sum() / len(rating_df)                 # mean squared error
    rmse_error = np.sqrt(mse_error)
    mae_error = abs(rating_df['rating'] - rating_df['prediction']).sum()/len(rating_df) 
    return mse_error, rmse_error, mae_error

def compute_score(test_reviews):
    """Look at 5% of most highly predicted books for each user.
    Return the average actual rating of those books.
    """
    # for each user
    g = test_reviews.groupby('user_id')

    # detect the top_5 movies as predicted by your algorithm
    top_5 = g.prediction.transform(
        lambda x: x >= x.quantile(.95)
    )

    # return the mean of the actual score on those
    return test_reviews.rating[top_5==1].mean()

class Baseline():
    """Fill in missing values using the global mean."""
    def __init__(self):
        self.global_mean = 0
        self.user_ids = None
        self.book_ids = None
        self.user_mean =  None
        self.book_mean = None

    def train(self, reviews):
        self.global_mean = reviews['rating'].sum()/len(reviews)
        self.user_mean = reviews.groupby('user_id').mean()['rating']
        self.book_mean = reviews.groupby('book_id').mean()['rating']
        self.user_ids = reviews.user_id.unique()
        self.book_ids = reviews.book_id.unique()

    def predict_one(self, user_id, book_id):
        if user_id in self.user_ids:
            if book_id in self.book_ids:
                prediction = self.user_mean.loc[user_id] + self.book_mean.loc[book_id] - self.global_mean
            else:
                prediction = self.user_mean.loc[user_id]
        else:
            if book_id in self.book_ids:
                prediction = self.book_mean.loc[book_id]
            else:
                prediction = self.global_mean
        return prediction
    def predict_all(self, reviews):
        reviews['prediction'] = np.zeros(len(reviews))
        for idx, i in reviews.iterrows():
            i['prediction'] = self.predict_one(i['user_id'], i['book_id'])

        return reviews

def build_baseline_model(rating_df):
    train_data, test_data = split_dataframe(rating_df)
    print('train_data mean rating: %.5f'%train_data.rating.mean())
    baseline_model = Baseline()
    baseline_model.train(train_data)
    train_reviews = baseline_model.predict_all(train_data)
    train_mse_error, train_rmse_error, train_mae_error = get_error(train_reviews)

    test_reviews = baseline_model.predict_all(test_data)
    test_mse_error, test_rmse_error, test_mae_error = get_error(test_reviews)

    evaluation_score = compute_score(test_reviews)

    print('train mse: %.5f \n test_mse: %.5f, test_rmse: %.5f, test_mae: %.5f, \n evaluation_score: %.5f'%(
        train_mse_error, test_mse_error, test_rmse_error, test_mae_error, evaluation_score))
    return train_mse_error, test_mse_error, test_rmse_error, test_mae_error, evaluation_score

