import argparse
import os
import random
import tensorflow as tf
import numpy as np
import pickle

parser = argparse.ArgumentParser(description="SOTS")
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-d", "--dataset", type=str, default="cifar10")
# ['cifar10','cifar100','fashion','stl_10','imdb','sarcasm']

parser.add_argument("-bs", "--batch_size", default=128, type=int)
parser.add_argument("-e", "--epochs", default=10, type=int)

results = parser.parse_args()

seed = results.seed
dataset = results.dataset
batch_size = results.batch_size
epochs = results.epochs

os.environ["PYTHONHASHSEED"] = str(seed)

random.seed(seed)

np.random.seed(seed)

tf.random.set_seed(seed)

os.environ["TF_DETERMINISTIC_OPS"] = "1"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import datasets
import models

X_train, y_train, X_test, y_test = datasets.get_data(dataset)
model = models.get_model(dataset)

print(y_train.shape)
print(y_test.shape)



model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    epochs=epochs,
)

from ChangeDatasize import get_model_softmax_scores

scores = get_model_softmax_scores(X_train, y_train, model)

name = str(dataset) + "_scores_s_" + str(seed) + "_bs_" + str(batch_size) + "_e_" + str(epochs) 

pickle.dump(scores, open(str(name) + ".p", "wb"))
