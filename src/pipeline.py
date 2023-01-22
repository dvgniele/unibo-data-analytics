from utils import get_data_from_csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def data_aquisition(path: str):
    return get_data_from_csv(path)


def data_visualization(data: pd.DataFrame):
    # data = {
    #     "a": np.arange(50),
    #     "c": np.random.randint(0, 50, 50),
    #     "d": np.random.randn(50),
    # }
    # data["b"] = data["a"] + 10 * np.random.randn(50)
    # data["d"] = np.abs(data["d"]) * 100

    print(data.loc[:, ["userId", "movieId", "rating"]])
    data.loc[:, "rating"] = np.array(data.loc[:, "rating"])

    movies = data.loc[:, ["movieId", "rating"]]
    print("len before: " + str(len(movies)))
    # print(movies)
    # movies.sort_values(by=["movieId"])
    # print(movies)
    movies.drop_duplicates(subset=None, keep="first", inplace=True)
    print("len after: " + str(len(movies)))

    # plt.scatter("movieId", "userId", s="rating", data=data_rip)
    # plt.xlabel("movieId")
    # plt.ylabel("userId")
    # plt.show()


def data_preprocessing():
    pass


def modeling():
    pass


def performance_evaluation():
    pass
