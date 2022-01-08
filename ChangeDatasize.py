import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
from sklearn.model_selection import train_test_split

# def get_train_data_by_scores(x, y, size, scores=None):
#     index = np.random.choice(range(0, len(x)), size, p=scores, replace=False)
#     Train_x = x[index]
#     Test_y = y[index]
#     return Train_x, Test_y

def _get_range_size(x, y):
    if not isinstance(x, list):
        range_size = len(x)
    elif not isinstance(y, list):
        range_size = len(y)
    else:
        range_size = len(x[0])
    return range_size

def _get_sub_x(x, index):
    Train_x = []
    
    if isinstance(x, list):
        for part_x in x:
            Train_x.append(part_x[index])
    else:
        Train_x = x[index]    
    
    return Train_x

def _get_concat_x(x, start, end):
    
    # np.concatenate(s_x[0 : i + 1])
    Train_x = []
    
    if isinstance(x[0], list):
        for i in range(len(x[0])):
            tmp = []
            for j in range(start, end):
                tmp.append(x[j][i])       
            Train_x.append(np.concatenate(tmp))
    else:
        Train_x = np.concatenate(x[start : end])
        
    return Train_x
    
def get_train_data_by_scores(x, y, size, scores=None):
    range_size = _get_range_size(x, y)
    index = np.random.choice(range(0, range_size), size, p=scores, replace=False)
    Train_x = _get_sub_x(x, index)
    Train_y = _get_sub_x(y, index)
    return Train_x, Train_y

def generic_shuffle_data(x, y):
    range_size = _get_range_size(x, y)
    index = np.random.choice(range(0, range_size), range_size, replace=False)
    Train_x = _get_sub_x(x, index)
    Train_y = _get_sub_x(y, index)
    return Train_x, Train_y

def get_cycle_data_sizes(
    size, start_percent, multiplier, end_percent, strategy, start_increase, total_sample
):
    sizes = []
    current_size = int(np.ceil(size * start_percent))
    sizes.append(current_size)
    max_size = int(size * max(end_percent, start_percent))
    min_size = int(np.ceil(size * min(end_percent, start_percent)))

    if strategy not in ["cycle", "lower_to_bigger", "bigger_to_lower", "sharp_cycle"]:
        raise Exception(
            "strategy must be one of these ['cycle','lower_to_bigger','bigger_to_lower','sharp_cycle']"
        )

    if strategy == "cycle":
        while sum(sizes) < total_sample:
            if start_increase:
                current_size = int(np.ceil(current_size * (1 / multiplier)))
                if current_size < max_size:
                    sizes.append(current_size)
                else:
                    sizes.append(max_size)
                    start_increase = False
            else:
                current_size = int(np.ceil(current_size * (multiplier)))
                if current_size > min_size:
                    sizes.append(current_size)
                else:
                    sizes.append(min_size)
                    start_increase = True
    if strategy == "lower_to_bigger":
        while sum(sizes) < total_sample:
            current_size = int(current_size * (1 / multiplier))
            if current_size < max_size:
                sizes.append(current_size)
            else:
                sizes.append(max_size)
    if strategy == "bigger_to_lower":
        while sum(sizes) < total_sample:
            current_size = int(current_size * (multiplier))
            if current_size > min_size:
                sizes.append(current_size)
            else:
                sizes.append(min_size)
    if strategy == "sharp_cycle":
        while sum(sizes) < total_sample:
            if start_increase:
                current_size = int(np.ceil(current_size * (1 / multiplier)))
                if current_size < max_size:
                    sizes.append(current_size)
                else:
                    sizes.append(max_size)
                    start_increase = False
            else:
                current_size = min_size
                sizes.append(current_size)
                start_increase = True

    return sizes


def get_cycle_sizes(sp, ep, alfa, T):
    S = []
    n = sp
    S.append(n)
    t = 0
    while sum(S) < T:
        if (n == sp) or ((S[t - 1] < S[t]) and (n != ep)):
            n = min((n * (1 / alfa)), ep)
        else:
            n = max((n * (alfa)), sp)
        S.append(n)
        t += 1
    return S


def get_lower_to_bigger_sizes(start, count, epochs, alfa):
    data_sizes = []
    # start = 0.22
    # count = 5
    current = start
    # epochs = epochs
    while sum(data_sizes) < epochs:
        for i in range(count):
            data_sizes.append(min(current, 1))
        current *= 1 / alfa
    return data_sizes


def get_bigger_to_lower_sizes(end, count, epochs, alfa):
    data_sizes = []
    # start = 0.22
    # count = 5
    current = 1
    # epochs = epochs
    while sum(data_sizes) < epochs:
        for i in range(count):
            data_sizes.append(max(current, end))
        current *= alfa
    return data_sizes

def get_sizes_from_weights(weights, epochs):
    sizes = []
    
    for k in range(len(weights)):
     
        s1 = sum(weights[:k+1]) / sum(weights)
        for i in range(epochs[k]):
            sizes.append(s1)
    
    return sizes
    
# def get_sizes(sp, alfa, T, episodes):


def get_model_sum_scores(x, y, model):
    if "sklearn" in str(type(model)):
        model.fit(x, np.argmax(y, axis=-1))
        p = model.predict_proba(x)
    elif "keras" in str(type(model)):
        p = model.predict(x)
    else:
        raise ValueError("Only Keras and Sklearn Models Supported")

    differance = abs(p - y)
    scores = np.mean(differance, axis=tuple(range(1, y.ndim)))

    return 1 - scores


def get_model_softmax_scores(x, y, model, batch_size = 32):
    if "sklearn" in str(type(model)):
        model.fit(x, np.argmax(y, axis=-1))
        p = model.predict_proba(x)
    elif "keras" in str(type(model)):
        p = model.predict(x, batch_size = batch_size)

    real_ind = np.argmax(y, axis=1)
    scores = p[np.arange(len(p)), real_ind]

    tf.keras.backend.clear_session()
    gc.collect()

    return scores


def get_model_softmax_score_p(x, y, model):
    if "sklearn" in str(type(model)):
        model.fit(x, np.argmax(y, axis=-1))
        p = model.predict_proba(x)
    elif "keras" in str(type(model)):
        p = model.predict(x)

    real_ind = np.argmax(y, axis=1)
    scores = p[np.arange(len(p)), real_ind]

    tf.keras.backend.clear_session()
    gc.collect()

    return scores, p


def load_scores(path):
    scores = pickle.load(open(path, "rb"))
    return scores

def get_ind_loss(model, X, Y, batch_size = 32):
    if np.array(Y).ndim == 1:
        return get_sparse_ind_loss(model, X, Y, batch_size = batch_size)
    else:
        return get_categ_ind_loss(model, X, Y, batch_size = batch_size)

def get_categ_ind_loss(model, X, Y, batch_size = 32):

    individual_loss_cal = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.compat.v1.losses.Reduction.NONE)
                
    pred = model.predict(X, batch_size, verbose = 0)
    ind_loss = individual_loss_cal(Y, pred).numpy()
    
    if len(ind_loss.shape) > 1:
        ind_loss = np.sum(ind_loss,axis = 1)
        
    tf.keras.backend.clear_session()
    gc.collect()

    return ind_loss

def get_sparse_ind_loss(model, X, Y, batch_size = 32):
    individual_loss_cal = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.compat.v1.losses.Reduction.NONE
    )
    pred = model.predict(X, batch_size, verbose = 0)
    ind_loss = individual_loss_cal(Y, pred).numpy()

    tf.keras.backend.clear_session()
    gc.collect()

    return ind_loss

def get_sub_data(x, y, percent = 0.5):
    (X_train, X_test, y_train, y_test) = train_test_split(x, y,  test_size = percent, random_state = 42, stratify = np.argmax(y, axis=1))
    
    return X_train, y_train

    
def get_sub_data_with_scores(x, y, scores, percent = 0.5):
    (X_train, X_test, y_train, y_test, scores_train, scores_test) = train_test_split(x, y, scores, test_size = percent, 
                                                               random_state = 42, stratify = np.argmax(y, axis=1))
    return X_train, y_train, scores_train






