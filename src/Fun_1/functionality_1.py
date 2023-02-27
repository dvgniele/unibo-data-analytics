import torch
from torch import nn
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error


class Bayes:
    def __init__(self, ratings, scores, seed = 1038893) -> None:
        movie_genome_scores = pd.merge(ratings, scores, on='movieId')

        x = movie_genome_scores.pivot(index='movieId', columns='tagId', values='relevance')
        y = ratings.groupby('movieId').mean()['rating']

        X_sparse = csr_matrix(x)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=seed)

        #*  Training
        nb = GaussianNB
        nb.fit(X_train.toarray(), y_train)

        #*  Testing
        y_pred = nb.predict(X_test.toarray())

        #*  Calculating Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)

        print(f'Mean Squared Error: {mse}')




class Trees:
    def __init__(self) -> None:
        pass


class KNN:
    def __init__(self) -> None:
        pass


class SVM:
    def __init__(self) -> None:
        pass
