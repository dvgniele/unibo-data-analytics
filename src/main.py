import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, Subset

import os

from utils import fix_random
from pipeline import data_aquisition, data_visualization


def main():
    seed = 27
    fix_random(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    #                                 #
    #                                 #

    ###################################

    #####     DATA AQUISITION     #####

    ###################################

    #                                 #
    #                                 #

    root = "./data/ml-25m"
    # root = "./locals"
    processed_path = ".data/merged.csv"

    genome_tags_path = "genome-tags.csv"
    ratings = "ratings.csv"

    data = None
    if os.path.exists(processed_path):
        data = data_aquisition(f"{processed_path}")
        print("Loaded merged data")
    else:
        data = data_aquisition(f"{root}/{ratings}")
        print(f"Loading ml-25m data: {root}/{ratings}")
        # print(data)

    #                                    #
    #                                    #

    ######################################

    #####     DATA VISUALIZATION     #####

    ######################################

    #                                    #
    #                                    #

    data_visualization(data)

    # genome_tags = data_aquisition([f"{root}/{genome_tags_path}"])
    # print(genome_tags)

    # hyperparameters
    num_epochs = 100  # try 100, 200, 500
    learning_rate = 0.01
    batch = 32

    hidden_size = 32
    dropout_prob = 0.2
    depth = 2


if __name__ == "__main__":
    main()
