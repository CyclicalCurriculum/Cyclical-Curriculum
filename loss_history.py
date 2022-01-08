import numpy as np
import gc
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

class BatchHistory(tf.keras.callbacks.Callback):
    def __init__(self, data=None, n=1, verbose=1, batch_size=1024):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.val_losses = []
        self.history = {"loss": [], "acc": []}
        self.data = data
        self.batch_count = 0
        self.n = n
        self.loss_cal = tf.keras.losses.CategoricalCrossentropy()
        self.acc_cal = tf.keras.metrics.CategoricalAccuracy()
        self.verbose = verbose
        self.batch_size = batch_size

    def on_batch_end(self, batch, logs=None):
        if self.batch_count % self.n == 0:

            pred = self.model.predict(self.data[0], self.batch_size)
            self.acc_cal.update_state(self.data[1], pred)
            acc = self.acc_cal.result().numpy()
            loss = self.loss_cal(self.data[1], pred).numpy()
            self.history["loss"].append(loss)
            self.history["acc"].append(acc)
            if self.verbose == 1:
                print(
                    " ",
                    self.batch_count,
                    ". Batch Loss: ",
                    "{:.4f}".format(loss),
                    " - Acc: ",
                    "{:.4f}".format(acc),
                    " - Max Acc: ",
                    "{:.4f}".format(max(self.history["acc"])),
                )

            self.acc_cal.reset_states()

        self.batch_count += 1
        tf.keras.backend.clear_session()
        gc.collect()


class EpochHistory(tf.keras.callbacks.Callback):
    def __init__(self, data=None, n=1, verbose=1, batch_size=1024):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.val_losses = []
        self.history = {"loss": [], "acc": []}
        self.data = data
        self.epoch_count = 0
        self.n = n
        self.loss_cal = tf.keras.losses.CategoricalCrossentropy()
        self.acc_cal = tf.keras.metrics.CategoricalAccuracy()
        self.verbose = verbose
        self.batch_size = batch_size

    def on_epoch_end(self, batch, logs=None):
        if self.epoch_count % self.n == 0:

            pred = self.model.predict(self.data[0], self.batch_size)
            self.acc_cal.update_state(self.data[1], pred)
            acc = self.acc_cal.result().numpy()
            loss = self.loss_cal(self.data[1], pred).numpy()
            self.history["loss"].append(loss)
            self.history["acc"].append(acc)
            if self.verbose == 1:
                print(
                    " ",
                    self.epoch_count,
                    ". Epoch Loss: ",
                    "{:.4f}".format(loss),
                    " - Acc: ",
                    "{:.4f}".format(acc),
                    " - Max Acc: ",
                    "{:.4f}".format(max(self.history["acc"])),
                )
                # print('f1_score:', classification_report(np.argmax(self.data[1], axis=-1), 
                #                             np.argmax(pred, axis=-1)))
            self.acc_cal.reset_states()

        self.epoch_count += 1
        tf.keras.backend.clear_session()
        gc.collect()


class WeightsBatchHistory(tf.keras.callbacks.Callback):
    def __init__(self, n=1):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.weights = []
        self.n = n
        self.epoch_count = 0

    def on_batch_end(self, epoch, logs=None):
        if self.epoch_count % self.n == 0:
            self.weights.append(self.model.get_weights()[-2:])
        self.epoch_count += 1
        tf.keras.backend.clear_session()
        gc.collect()

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_count == 0:
            self.weights.append(self.model.get_weights()[-2:])


class WeightsEpochHistory(tf.keras.callbacks.Callback):
    def __init__(self, n=1):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.weights = []
        self.n = n
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_count % self.n == 0:
            self.weights.append(self.model.get_weights()[-2:])
        self.epoch_count += 1
        tf.keras.backend.clear_session()
        gc.collect()

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_count == 0:
            self.weights.append(self.model.get_weights()[-2:])

# class Metrics(tf.keras.callbacks.Callback):
#     def __init__(self, data=None, n=1, verbose=1, batch_size=1024):
#         super(tf.keras.callbacks.Callback, self).__init__()
#         self.n = n
#         self.epoch_count = 0

#     def on_epoch_end(self, epoch, logs=None):
#         if epoch:
#             print(self.data[0])
#             x_test = self.data[0]
#             y_test = self.validation_data[1]
#             predictions = self.model.predict(x_test)
#             print('f1_score:', f1_score(np.argmax(y_test, axis=-1), 
#                                         np.argmax(predictions, axis=-1)).round(2))
#         self.epoch_count += 1
#         tf.keras.backend.clear_session()
#         gc.collect()

#     def on_epoch_begin(self, epoch, logs=None):
#         if self.epoch_count == 0:
#             self.weights.append(self.model.get_weights())


class Without_O_Accuracy(tf.keras.callbacks.Callback):
    def __init__(self,data):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.without_o_acc = []
        self.data = data
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.data[0], self.data[1]
        acc = self.acc_without_o(X_val,y_val)
        self.without_o_acc.append(acc)
        print(self.epoch_count, ". Epoch Acc:","{:.4f}".format(acc), "Max Acc", "{:.4f}".format(max(self.without_o_acc)))
        # print(acc1)
        self.epoch_count += 1
        
    def acc_without_o(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=2)
        y_true = np.argmax(y_test, axis=2)

        no_O_indices = np.where(y_true != 2)
        y_pred_without_O = y_pred[no_O_indices]
        y_true_without_O = y_true[no_O_indices]
        
        true = (y_true_without_O == y_pred_without_O).sum()
        size = y_true_without_O.size
        
        acc = true / size
    
        return acc
    def acc_without_o2(self, x_test, y_test, un):
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=2)
        y_true = np.argmax(y_test, axis=2)
    
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
    
        no_O_indices = np.where(y_true != un)
    
        y_pred_without_O = y_pred[no_O_indices]
        y_true_without_O = y_true[no_O_indices]
    
        sum_total = 0
        len_total = len(y_true_without_O)
    
        for i in range(len(y_true_without_O)):
            if y_pred_without_O[i] == y_true_without_O[i]:
                sum_total += 1
    
        acc = sum_total / len_total
        return acc



class DynamicLossEpochHistory(tf.keras.callbacks.Callback):
    def __init__(self, data=None, n=1, verbose=1, batch_size=1024, GAMA=0.90):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.val_losses = []
        self.history = {
            "loss": [],
            "acc": [],
            "ind_loss": [],
            "DIH": [],
            "dDIH": [],
            "true_preds": [],
            "label_change": [],
            "scores": [],
        }
        self.data = data
        self.epoch_count = 0
        self.n = n
        self.loss_cal = tf.keras.losses.CategoricalCrossentropy()
        self.individual_loss_cal = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.compat.v1.losses.Reduction.NONE
        )
        self.acc_cal = tf.keras.metrics.CategoricalAccuracy()
        self.verbose = verbose
        self.batch_size = batch_size
        self.GAMA = GAMA
        self.DIH = np.zeros(len(data[0]))
        self.dDIH = np.zeros(len(data[0]))
        self.ind_val_loss = np.zeros(len(data[0]))
        self.true_label = np.argmax(self.data[1], axis=-1)
        self.label_pred = np.zeros(len(data[0]))

    def get_losses(self, pred):
        self.acc_cal.update_state(self.data[1], pred)
        val_acc = self.acc_cal.result().numpy()
        val_loss = self.loss_cal(self.data[1], pred).numpy()
        self.history["loss"].append(val_loss)
        self.history["acc"].append(val_acc)
        self.acc_cal.reset_states()
        return val_acc, val_loss

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_count % self.n == 0:

            prev_ind_loss = self.ind_val_loss.copy()
            prev_label_pred = self.label_pred.copy()

            pred = self.model.predict(self.data[0], self.batch_size)
            val_acc, val_loss = self.get_losses(pred)
            self.ind_val_loss = self.individual_loss_cal(self.data[1], pred).numpy()
            self.DIH = (self.GAMA * self.ind_val_loss) + ((1 - self.GAMA) * self.DIH)
            dLoss = abs(prev_ind_loss - self.ind_val_loss)
            self.dDIH = (self.GAMA * dLoss) + ((1 - self.GAMA) * self.dDIH)
            self.history["ind_loss"].append(self.ind_val_loss)
            self.history["DIH"].append(self.DIH)
            self.history["dDIH"].append(self.dDIH)
            self.label_pred = np.argmax(pred, axis=-1)
            true_preds = np.where(self.true_label == self.label_pred, 1.0, 0.0)
            self.history["true_preds"].append(true_preds)
            label_change = np.where(prev_label_pred == self.label_pred, 0, 1)
            self.history["label_change"].append(label_change)

            self.history["scores"].append(pred[np.arange(len(pred)), self.true_label])

        self.epoch_count += 1
        tf.keras.backend.clear_session()
        gc.collect()
