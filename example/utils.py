import tensorflow as tf
import gc
import numpy as np

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

def get_train_data_by_scores(x, y, size, scores=None):
    range_size = _get_range_size(x, y)
    index = np.random.choice(range(0, range_size), size, p=scores, replace=False)
    Train_x = _get_sub_x(x, index)
    Train_y = _get_sub_x(y, index)
    return Train_x, Train_y

def CyclicalTrain(
    model,
    x,
    y,
    data_sizes,
    scores=None,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=1,
    data=None,
):
    total_sample_count = _get_range_size(x, y)
    current_max = 0
    val_accs = []
    train_accs = []
    val_losses = []
    train_losses = []
    result_dict = {}
    epochs = len(data_sizes)
    for i in range(epochs):
        sample_count_epoch = int(total_sample_count * data_sizes[i])
        sub_x, sub_y = get_train_data_by_scores(
            x, y, sample_count_epoch, scores=scores / scores.sum()
        )
        
        history = model.fit(
            sub_x,
            sub_y,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=data,
        )
        current = history.history["val_accuracy"][0]
        current_max = max(current_max, current)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        
        val_accs.append(current)
        train_accs.append(history.history["accuracy"][0])
        val_losses.append(history.history["val_loss"][0])
        train_losses.append(history.history["loss"][0])        
        
        tf.keras.backend.clear_session()
        gc.collect()
    
    result_dict['accuracy'] = train_accs
    result_dict['loss'] = train_losses
    result_dict['val_accuracy'] = val_accs
    result_dict['val_loss'] = val_losses
    
    return model, current_max, result_dict

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
