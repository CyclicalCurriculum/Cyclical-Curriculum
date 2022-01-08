import numpy as np
import keras

import time

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from pacing_functions import exponential_pacing_normalized, parabolic_pacing_normalized

from sklearn.utils import shuffle

import pickle

def _get_sub_x(x, index):
    Train_x = []
    
    if isinstance(x, list):
        for part_x in x:
            Train_x.append(part_x[index])
    else:
        Train_x = x[index]    
    
    return Train_x

def _get_range_size(x, y):
    if not isinstance(x, list):
        range_size = len(x)
    elif not isinstance(y, list):
        range_size = len(y)
    else:
        range_size = len(x[0])
    return range_size

def split_data_x_y(X, Y, N, arr_ind=[], weights=[]):

    if len(weights) == 0:
        weights = [1] * N

    if len(arr_ind) == 0:
        arr_ind = list(range(_get_range_size(X,Y)))
        np.random.shuffle(arr_ind)

    new_weighs = []

    total = sum(weights)

    for i in range(len(weights)):
        t = int((sum(weights[: i + 1]) / total) * _get_range_size(X,Y))
        new_weighs.append(t)

    splited_x = []
    splited_y = []
    
    # return arr_ind, new_weighs[:-1]
    
    for i in np.array_split(arr_ind, new_weighs[:-1]):
        # print('burda')
        # print(i)
        splited_x.append(_get_sub_x(X,i))
        splited_y.append(_get_sub_x(Y,i))

    return splited_x, splited_y


def split_data_x(X, N, arr_ind=[], weights=[]):

    if len(weights) == 0:
        weights = [1] * N

    if len(arr_ind) == 0:
        arr_ind = list(range(len(X)))
        np.random.shuffle(arr_ind)

    new_weighs = []

    total = sum(weights)

    for i in range(len(weights)):
        t = int((sum(weights[: i + 1]) / total) * len(X))
        new_weighs.append(t)

    splited_x = []

    for i in np.array_split(arr_ind, new_weighs[:-1]):
        splited_x.append(X[i])

    return splited_x

def split_data_x_y_z(X, Y, Z, N, arr_ind=[], weights=[]):

    if len(weights) == 0:
        weights = [1] * N

    if len(arr_ind) == 0:
        arr_ind = list(range(len(X)))
        np.random.shuffle(arr_ind)

    new_weighs = []

    total = sum(weights)

    for i in range(len(weights)):
        t = int((sum(weights[: i + 1]) / total) * len(X))
        new_weighs.append(t)

    splited_x = []
    splited_y = []
    splited_z = []

    for i in np.array_split(arr_ind, new_weighs[:-1]):
        splited_x.append(X[i])
        splited_y.append(Y[i])
        splited_z.append(Z[i])

    return splited_x, splited_y, splited_z


def get_model_scores(x, y, model):
    if y.ndim < 3:
        return get_model_softmax_scores(x, y, model)
    else:
        return get_model_sum_scores(x, y, model)


def get_model_sum_scores(x, y, model):
    if "sklearn" in str(type(model)):
        model.fit(x, np.argmax(y, axis=-1))
        p = model.predict_proba(x)
    elif "keras" in str(type(model)):
        p = model.predict(x)
    else:
        raise ValueError("Only Keras and Sklearn Models Supported")

    differance = abs(p - y)
    scores = np.sum(differance, axis=tuple(range(1, y.ndim)))

    return 1 / scores


def get_model_softmax_scores(x, y, model):
    if "sklearn" in str(type(model)):
        model.fit(x, np.argmax(y, axis=-1))
        p = model.predict_proba(x)
    elif "keras" in str(type(model)):
        p = model.predict(x)

    real_ind = np.argmax(y, axis=1)
    scores = p[np.arange(len(p)), real_ind]

    return scores


def get_split_data_by_weights(x, Y, N, model, weights, is_curriculum=True):

    scores = get_model_scores(x, Y, model)

    if is_curriculum == False:
        scores *= -1
    if str(is_curriculum) == "mixed1":
        scores = abs(scores - 0.5)
    if str(is_curriculum) == "mixed2":
        scores = abs(scores - 0.5)
        scores *= -1

    if Y.ndim < 3:

        final_indices = get_class_balanced_indies(scores, Y, Y.shape[1], N, weights)

        splited_x1, splited_y1 = split_data_x_y(
            x, Y, N, arr_ind=final_indices, weights=weights
        )

    else:
        splited_x1, splited_y1 = split_data_x_y(
            x, Y, N, arr_ind=scores.argsort()[::-1], weights=weights
        )

    return splited_x1, splited_y1


def get_class_balanced_indies(scores, y, num_classes, N, weights):
    
    if not isinstance(y, list) and y.ndim < 3 and y.ndim > 1:
    
        y = np.argmax(y, axis=-1)
    
        a = np.concatenate((scores.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    
        class_by_class_sorted = []
    
        for i in range(num_classes):
    
            d = a[a[:, 1] == i][:, 0].argsort()[::-1]
    
            b = np.where(a[:, 1] == i)[0]
    
            e = b[d]
            class_by_class_sorted.append(split_data_x(e, N, list(range(len(e))), weights))
    
        class_by_class_sorted = np.array(class_by_class_sorted, dtype="object")
    
        final_inds = []
    
        for i in range(N):
            final_inds += np.concatenate(class_by_class_sorted[:, i]).tolist()
    
    else:
        final_inds = scores.argsort()
    
    return final_inds


def fit_self_thougt(x, y, model, epochs, batch_size=32):
    new_model = keras.models.clone_model(model)
    new_model.compile(model.optimizer, model.loss, model.metrics_names[1:])
    new_model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return new_model


def get_split_data_by_function(
    x,
    y,
    model,
    is_curriculum,
    batch_size=32,
    function="parabolic_pacing",
    start_percent=0.04,
):
    weights = _get_weights_for_function(
        x, pacing=function, batch_size=batch_size, starting_percent=start_percent
    )
    n = len(weights)
    return get_split_data_by_weights(x, y, n, model, weights, is_curriculum)


def get_split_data_self_though_by_weights(
    x, y, n, model, weights, is_curriculum, epochs=1, batch_size=32
):
    new_model = fit_self_thougt(x, y, model, epochs, batch_size=batch_size)
    return get_split_data_by_weights(x, y, n, new_model, weights, is_curriculum=True)


def get_split_data_self_though_by_function(
    x, y, model, function, start_percent, is_curriculum, epochs=1, batch_size=32
):
    new_model = fit_self_thougt(x, y, model, epochs, batch_size=batch_size)
    weights = _get_weights_for_function(
        x, function, batch_size=batch_size, starting_percent=start_percent
    )
    n = len(weights)
    return get_split_data_by_weights(x, y, n, new_model, weights, is_curriculum)


def _get_weights_for_function(
    x, pacing, batch_size, increase_amount=2, starting_percent=0.04
):

    if pacing == "exponential_pacing":
        weights = exponential_pacing_normalized(
            len(x),
            batch_size,
            increase_amount=increase_amount,
            starting_percent=starting_percent,
        )
    elif pacing == "parabolic_pacing":
        weights = parabolic_pacing_normalized(
            len(x), batch_size, power=increase_amount, starting_percent=starting_percent
        )
    else:
        raise ValueError("exponential_pacing or parabolic_pacing available")

    return weights


def get_regression_sort(x, y, n, model, weights, is_curriculum=True):
    y_pred = model.predict(x)
    y_pred = y_pred.reshape(
        y_pred.shape[0],
    )

    scores = (y - y_pred) ** 2
    if is_curriculum:
        final_inds = np.argsort(scores)
    if is_curriculum == False:
        final_inds = np.argsort(scores)[::-1]
    if is_curriculum == "mixed1":
        mean = np.mean(scores)
        scores = abs(scores - mean)
        final_inds = np.argsort(scores)
    if is_curriculum == "mixed2":
        mean = np.mean(scores)
        scores = abs(scores - mean)
        final_inds = np.argsort(scores)[::-1]

    splited_x1, splited_y1 = split_data_x_y(
        x, y, n, arr_ind=final_inds, weights=weights
    )

    return splited_x1, splited_y1


def get_random_split_data_by_function(
    x, y, batch_size=32, function="parabolic_pacing", start_percent=0.04
):
    weights = _get_weights_for_function(
        x, pacing=function, batch_size=batch_size, starting_percent=start_percent
    )
    n = len(weights)
    splitted_x, splitted_y = split_data_x_y(x, y, n, weights=weights)
    return splitted_x, splitted_y


def load_scores(path):
    scores = pickle.load(open(path, "rb"))
    return scores
