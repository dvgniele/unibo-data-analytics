import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, Subset

from utils import fix_random

from functionality_1 import Bayes, Trees, KNN, SVM
from functionality_2 import NeuralNetwork, DeepNeuralNetwork
from functionality_3 import DeepForTabularData_1, DeepForTabularData_2

def main():
    seed = 27
    fix_random(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    #hyperparameters
    num_epochs = 100  # try 100, 200, 500
    learning_rate = 0.01
    batch = 32

    hidden_size = 32
    dropout_prob = 0.2
    depth = 2


if __name__ == "__main__":
    main()

print(torch.cuda.is_available())