import numpy as np

import argparse
import pickle
import random
import os
import tensorflow as tf

parser = argparse.ArgumentParser(description="SOTS")
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("-d", "--dataset", type=str, default="cifar10")
# ['cifar10','cifar100','fashion','stl_10','imdb','sarcasm']

parser.add_argument("-bs", "--batch_size", default=128, type=int)
parser.add_argument("-e", "--epochs", default=10, type=int)

parser.add_argument("-sp", "--start_percent", default=0.25, type=float)
parser.add_argument("-m", "--multiplier", default=0.5, type=float)
parser.add_argument("-ep", "--end_percent", default=1.0, type=float)
parser.add_argument("-st", "--strategy", default="cycle", type=str)
parser.add_argument("-si", "--start_increase", default=True, type=bool)
parser.add_argument("-sc", "--sample_count", default=300, type=int)
parser.add_argument("-ct", "--curriculum_type", default='easy', type=str)

parser.add_argument("--min_lr", default=0.01, type=float)
parser.add_argument("--max_lr", default=0.01, type=float)
parser.add_argument("--lr_multiplier", default=1.0, type=float)
parser.add_argument("--lr_decay", default=1.0, type=float)

parser.add_argument("-spath", "--score_path", default="", type=str)

results = parser.parse_args()
print(results)

seed = results.seed
dataset = results.dataset
batch_size = results.batch_size
epochs = results.epochs
start_percent = results.start_percent
multiplier = results.multiplier
end_percent = results.end_percent
strategy = results.strategy
start_increase = results.start_increase
sample_count = results.sample_count
min_lr = results.min_lr
max_lr = results.max_lr
lr_multiplier = results.lr_multiplier
lr_decay = results.lr_decay
score_path = results.score_path
curriculum_type = results.curriculum_type



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

from ChangeDatasize import (
    get_cycle_data_sizes,
    load_scores,
)

if curriculum_type == 'random':
    # print('girdi')
    scores = np.ones(len(X_train))
else:
    scores = load_scores(score_path)
    if curriculum_type == 'easy':
        scores = scores
    elif curriculum_type == 'hard':
        scores = 1 / scores
    elif curriculum_type == 'mediocre':
        scores = 1 - abs(scores - 0.5)
    elif curriculum_type == 'reverse_mediocre':
        scores = abs(scores - 0.5)



train_data_size = len(X_train)
total_sample = train_data_size * epochs

cycle_step = np.ceil(train_data_size / batch_size) * 2

n = np.ceil(np.ceil(train_data_size / batch_size) * epochs / 300)

# print(np.ceil(train_data_size / batch_size))
# print(np.ceil((train_data_size / 2) / batch_size))
# print(np.ceil((train_data_size / 4) / batch_size))
# print(n)

print(len(X_train))
print(len(X_test))

# 1 / 0

from loss_history import BatchHistory

h = BatchHistory(val_data=(X_train, y_train), n=n)

from CycleLearningRate import SGDRScheduler

from CycleLearningRate2 import CyclicLR

# clr = SGDRScheduler(
#     min_lr=min_lr,
#     max_lr=max_lr,
#     cycle_step=cycle_step,
#     lr_decay=lr_decay,
#     mult_factor=lr_multiplier,
# )

clr = CyclicLR(base_lr=0.001, max_lr=0.006,
               step_size=2000., mode='triangular')

data_sizes = get_cycle_data_sizes(
    size=train_data_size,
    start_percent=start_percent,
    multiplier=multiplier,
    end_percent=end_percent,
    strategy=strategy,
    start_increase=start_increase,
    total_sample=total_sample,
)

from DataSizeTrain import ChangingDatasizeTrain

ChangingDatasizeTrain(
    model,
    X_train,
    y_train,
    data_sizes,
    scores=scores,
    batch_size=batch_size,
    epochs=len(data_sizes),
    callbacks=[h],
)

name = (
    str(dataset)
    + '_'
    + str(curriculum_type)
    + "_s_"
    + str(seed)
    + "_bs_"
    + str(batch_size)
    + "_e_"
    + str(epochs)
    + "_sp_"
    + str(start_percent)
    + "_m_"
    + str(multiplier)
    + "_ep_"
    + str(end_percent)
    + "_st_"
    + str(strategy)
    + "_i_"
    + str(start_increase)
    + "_sc_"
    + str(sample_count)
    + "_lr_"
    + str(str(min_lr) + "_" + str(max_lr) + "_" + str(lr_multiplier) + "_" + str(lr_decay))
    + "_sc_"
    + str(score_path)
)

name += ".p"

pickle.dump(h.history["val_acc"], open(name, "wb"))
