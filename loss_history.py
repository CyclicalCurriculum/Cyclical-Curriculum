import numpy as np
import keras
import gc

from pacing_functions import *

import time

# # Seed value
# # Apparently you may use different seed values at each stage
# seed_value= 42

# # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)

# # 2. Set `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)

# # 3. Set `numpy` pseudo-random generator at a fixed value
# import numpy as np
# np.random.seed(seed_value)

# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)

from keras import backend as K


import matplotlib.pyplot as plt
import keras


import tensorflow as tf
from sklearn.metrics import accuracy_score

from keras.utils import to_categorical


class BatchHistory(tf.keras.callbacks.Callback):
    def __init__(self, val_data=None, train_data=None, n=1, verbose=1,batch_size = 1024):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.val_losses = []
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        self.val_data = val_data
        self.train_data = train_data
        self.batch_count = 0
        self.n = n
        self.loss_cal = tf.keras.losses.CategoricalCrossentropy()
        self.acc_cal = tf.keras.metrics.CategoricalAccuracy()
        self.verbose = verbose
        self.batch_size = batch_size

    def on_batch_end(self, batch, logs=None):
        if self.batch_count % self.n == 0:
            if self.val_data != None:
                pred = self.model.predict(self.val_data[0],self.batch_size)
                self.acc_cal.update_state(self.val_data[1], pred)
                val_accuracy = self.acc_cal.result().numpy()
                val_loss = self.loss_cal(self.val_data[1], pred).numpy()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_accuracy)
                if self.verbose == 1:
                    print(
                        " ",
                        self.batch_count,
                        ". Batch Val Loss: ",
                        np.round(val_loss, 4),
                        "Val Acc: ",
                        np.round(val_accuracy, 4),
                        "Max Val Acc: ",
                        np.round(max(self.history["val_acc"]), 4),
                    )

                self.acc_cal.reset_states()

            if self.train_data != None:
                pred = self.model.predict(self.train_data[0],self.batch_size)
                self.acc_cal.update_state(self.train_data[1], pred)
                accuracy = self.acc_cal.result().numpy()
                loss = self.loss_cal(self.train_data[1], pred).numpy()
                if self.verbose == 1:
                    print(
                        " ",
                        self.batch_count,
                        ". Batch Training Loss: ",
                        np.round(loss, 4),
                        "Training Acc: ",
                        np.round(accuracy, 4),
                    )
                self.history["loss"].append(loss)
                self.history["acc"].append(accuracy)
                self.acc_cal.reset_states()

        self.batch_count += 1
        tf.keras.backend.clear_session()
        gc.collect()


class EpochHistory(tf.keras.callbacks.Callback):
    def __init__(self, val_data=None, train_data=None, n=1, verbose=1,batch_size = 1024):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.val_losses = []
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        self.val_data = val_data
        self.train_data = train_data
        self.epoch_count = 0
        self.n = n
        self.loss_cal = tf.keras.losses.CategoricalCrossentropy()
        self.acc_cal = tf.keras.metrics.CategoricalAccuracy()
        self.verbose = verbose
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_count % self.n == 0:
            if self.val_data != None:
                pred = self.model.predict(self.val_data[0],self.batch_size)
                self.acc_cal.update_state(self.val_data[1], pred)
                val_accuracy = self.acc_cal.result().numpy()
                val_loss = self.loss_cal(self.val_data[1], pred).numpy()
                if self.verbose == 1:
                    print(
                        " ",
                        self.epoch_count,
                        ".Epoch Val Loss: ",
                        np.round(val_loss, 4),
                        "Val Acc: ",
                        np.round(val_accuracy, 4),
                        "Max Val Acc: ",
                        np.round(max(val_accuracy), 4),
                    )
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_accuracy)
                self.acc_cal.reset_states()

            if self.train_data != None:
                pred = self.model.predict(self.train_data[0],self.batch_size)
                self.acc_cal.update_state(self.train_data[1], pred)
                accuracy = self.acc_cal.result().numpy()
                loss = self.loss_cal(self.train_data[1], pred).numpy()
                if self.verbose == 1:
                    print(
                        " ",
                        self.epoch_count,
                        ".Epoch Training Loss: ",
                        np.round(loss, 4),
                        "Training Acc: ",
                        np.round(accuracy, 4),
                    )
                self.history["loss"].append(loss)
                self.history["acc"].append(accuracy)
                self.acc_cal.reset_states()

        self.epoch_count += 1
        tf.keras.backend.clear_session()
        gc.collect()


# class EveryBatchAllTrainingAndValidationHistory(keras.callbacks.Callback):
#     def __init__(self , val_data = None, train_data = None):
#         super(keras.callbacks.Callback, self).__init__()
#         self.losses = []
#         self.val_losses = []
#         self.history = {'loss':[],'val_loss':[],'acc':[],'val_acc':[]}
#         self.val_data = val_data
#         self.train_data = train_data
#         self.k = 0
#         self.switch_points = []
#     # def on_train_begin(self, logs=None):
#     #     self.losses = []
#     #     self.val_losses = []

#     def on_batch_end(self, batch, logs=None):
#         if self.k % 10 == 0:
#             # print('----------------------')
#             # print(len(self.validation_data[0]))
#             # print(len(self.training_data[0]))
#             if self.val_data != None:
#                 val_evaluation = self.model.evaluate(self.val_data[0], self.val_data[1],verbose = 0)
#                 val_loss = val_evaluation[0]
#                 val_accuracy = val_evaluation[1]
#                 # print()
#                 # print('val_loss', val_loss, 'val_accuracy', val_accuracy)
#                 self.history['val_loss'].append(val_loss)
#                 self.history['val_acc'].append(val_accuracy)
#             if self.train_data != None:
#                 training_evaluation = self.model.evaluate(self.train_data[0], self.train_data[1],verbose = 0)
#                 tr_loss = training_evaluation[0]
#                 tr_accuracy = training_evaluation[1]
#                 self.history['loss'].append(tr_loss)
#                 self.history['acc'].append(tr_accuracy)
#             # self.val_losses.append(self.model)
#         else:
#             pass
#         self.k += 1
#     def on_epoch_end(self, epoch, logs=None):
#         if self.k % 1 == 0:
#             # print('----------------------')
#             if self.val_data != None:
#                 val_evaluation = self.model.evaluate(self.val_data[0], self.val_data[1],verbose = 0)
#                 val_loss = val_evaluation[0]
#                 val_accuracy = val_evaluation[1]
#                 print('val_loss', val_loss, 'val_accuracy', val_accuracy)
#                 self.history['val_loss'].append(val_loss)
#                 self.history['val_acc'].append(val_accuracy)
#             if self.train_data != None:
#                 training_evaluation = self.model.evaluate(self.train_data[0], self.train_data[1],verbose = 0)
#                 tr_loss = training_evaluation[0]
#                 tr_accuracy = training_evaluation[1]
#                 self.history['loss'].append(tr_loss)
#                 self.history['acc'].append(tr_accuracy)
#             # self.val_losses.append(self.model)
#         else:
#             pass
#         self.k += 1


# class EveryEpochAllTrainingAndValidationHistory(keras.callbacks.Callback):
#     def __init__(self , val_data = None, train_data = None):
#         super(keras.callbacks.Callback, self).__init__()
#         self.losses = []
#         self.val_losses = []
#         self.history = {'loss':[],'val_loss':[],'acc':[],'val_acc':[]}
#         self.val_data = val_data
#         self.train_data = train_data
#         self.k = 0
#         self.switch_points = []
#     # def on_train_begin(self, logs=None):
#     #     self.losses = []
#     #     self.val_losses = []

#     def on_epoch_end(self, epoch, logs=None):
#         if self.k % 1 == 0:
#             # print('----------------------')
#             if self.val_data != None:
#                 val_evaluation = self.model.evaluate(self.val_data[0], self.val_data[1],verbose = 0)
#                 val_loss = val_evaluation[0]
#                 val_accuracy = val_evaluation[1]
#                 print('val_loss', val_loss, 'val_accuracy', val_accuracy)
#                 self.history['val_loss'].append(val_loss)
#                 self.history['val_acc'].append(val_accuracy)
#             if self.train_data != None:
#                 training_evaluation = self.model.evaluate(self.train_data[0], self.train_data[1],verbose = 0)
#                 tr_loss = training_evaluation[0]
#                 tr_accuracy = training_evaluation[1]
#                 self.history['loss'].append(tr_loss)
#                 self.history['acc'].append(tr_accuracy)
#             # self.val_losses.append(self.model)
#         else:
#             pass
#         self.k += 1


# def on_epoch_end(self,batch,logs = None):
#     print('----------------------')
#     self.losses.append(logs.get('loss'))
#     evaluation = self.model.evaluate(self.validation_data[0], self.validation_data[1],verbose = 0)
#     val_loss = evaluation[0]
#     val_accuracy = evaluation[1]
#     print(val_loss)
#     self.history['val_loss'].append(val_loss)

# history = LossHistory()


# model = keras.Sequential()
# model.add(keras.layers.Dense(32, activation='relu', input_dim=100))
# model.add(keras.layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# # Generate dummy data
# import numpy as np
# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000, 1))

# data2 = np.random.random((1000, 100))
# labels2 = np.random.randint(2, size=(1000, 1))

# history2 = Histories((data2,labels2))

# # Train the model, iterating on the data in batches of 32 samples
# model.fit(data, labels, epochs=10, batch_size=32,
#           validation_data = (data2,labels2), callbacks=[history])

# # Plot the history
# y1=history.history['loss']
# y2=history.history['val_loss']  # val_loss nonetype on batch endde validationu alamıyor.

# # x1 = np.arange( len(y1))
# # k=len(y1)/len(y2)
# # x2 = np.arange(k,len(y1)+1,k)
# # fig, ax = plt.subplots()
# # line1, = ax.plot(x1, y1, label='loss')
# # line2, = ax.plot(x2, y2, label='val_loss')
# # plt.show()
