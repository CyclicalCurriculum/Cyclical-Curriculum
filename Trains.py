import gc
import numpy as np
import tensorflow as tf
from ChangeDatasize import get_train_data_by_scores, get_model_softmax_scores, _get_range_size, _get_concat_x



def AgingScoredCyclicalTrain(
    model,
    x,
    y,
    data_sizes,
    scores=None,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    reduced=1.2,
):

    first_scores = scores.copy()

    for i in range(epochs):

        index = np.random.choice(
            range(0, len(x)), data_sizes[i], p=scores / scores.sum(), replace=False
        )

        model.fit(
            x[index],
            y[index],
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        scores = first_scores.copy()

        scores[index] /= reduced
        scores[index] /= reduced

        tf.keras.backend.clear_session()
        gc.collect()


def Cyclical_plus_RandomMixupTrain(
    model,
    x,
    y,
    data_sizes,
    scores=None,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    val_data=None,
    alpha=0.6,
):
    total_sample_count = len(x)
    for i in range(epochs):
        sample_count_epoch = int(total_sample_count * data_sizes[i])
        sub_x, sub_y = get_train_data_by_scores(
            x, y, sample_count_epoch, scores=scores / scores.sum()
        )
        mixupX, mixupY = get_random_mixup(x, y, x, y, 1.0, alpha=alpha)

        model.fit(
            np.concatenate((sub_x, mixupX)),
            np.concatenate((sub_y, mixupY)),
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_data,
        )

        tf.keras.backend.clear_session()
        gc.collect()

    return model


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
        # "{:.4f}".format(max(self.history["acc"]))
        # print('Current Mx Val Acc', current_max)
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


def CyclicalTrainSLP(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    data=None,
    
):
    # total_sample_count = len(x)
    total_sample_count = _get_range_size(x, y)
    current_max = 0
    val_accs = []
    train_accs = []
    val_losses = []
    train_losses = []
    result_dict = {}
    for i in range(epochs):
        sample_count_epoch = int(total_sample_count * data_sizes[i])
        # scores = get_model_softmax_scores(x, y, model)
        scores = get_ind_loss(model, x, y, batch_size)
        scores = 1 / scores
        
        
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
        # "{:.4f}".format(max(self.history["acc"]))
        # print('Current Mx Val Acc', current_max)
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


def RandomMixupTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    alpha=0.4,
    val_data=None,
):
    current_max = 0
    patience_threshold = 10
    patiance = 0
    for i in range(epochs):

        mixupX, mixupY = get_random_mixup(x, y, x, y, data_sizes[i], alpha=alpha)
        # mixupX, mixupY = get_random_mixup(mixupX, mixupY, mixupX, mixupY, data_sizes[i], alpha = alpha)

        history = model.fit(
            mixupX,
            mixupY,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_data,
        )
        current = history.history["val_accuracy"][0]
        # current_max = max(current_max, current)
        # # "{:.4f}".format(max(self.history["acc"]))
        # # print('Current Mx Val Acc', current_max)
        # # print("Current Max Val Acc", "{:.4f}".format(current_max))

        if current > current_max:
            patiance = 0
            current_max = current
        else:
            patiance += 1
        
        # print(patiance)
        
        # if round(current,4) < round(current_max,4):
        #     patiance = 0
        # else:
        #     patiance += 1
              
        if patiance >= patience_threshold or round(current_max,4) >= 1:
            break

        
        
        
        
        tf.keras.backend.clear_session()
        gc.collect()

    return model, current_max


def CurriculumMixupTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    anti=False,
    alpha=0.4,
):
    for i in range(epochs):

        scores = get_ind_loss(model, x, y)

        if anti == False:
            scores = 1 / scores
        else:
            scores = scores

        sub_x1, sub_y1 = get_train_data_by_scores(
            x, y, int(len(x) * data_sizes[i]), scores=scores / scores.sum()
        )

        sub_x2, sub_y2 = get_train_data_by_scores(
            x, y, int(len(x) * data_sizes[i]), scores=scores / scores.sum()
        )

        mixupX, mixupY = get_random_mixup(sub_x1, sub_y1, sub_x2, sub_y2, alpha=alpha)

        model.fit(
            mixupX,
            mixupY,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        tf.keras.backend.clear_session()
        gc.collect()

    return model

def ProbabilisticCurriculumTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    ct_type="False",
    val_data = None
):
    current_max = 0
    for i in range(epochs):

        scores = get_ind_loss(model, x, y)

        if ct_type == "easy":
            scores = 1 / scores
        elif ct_type == "hard":
            scores = scores
        elif ct_type == "random":
            scores = np.ones(len(y))


        sub_x, sub_y = get_train_data_by_scores(
            x, y, int(len(x) * data_sizes[i]), scores=scores / scores.sum()
        )

        history = model.fit(
             sub_x,
             sub_y,
             epochs=1,
             batch_size=batch_size,
             verbose=verbose,
             callbacks=callbacks,
             validation_data = val_data
         )
        
        current = history.history["val_accuracy"][0]
        current_max = max(current_max, current)
        # "{:.4f}".format(max(self.history["acc"]))
        # print('Current Mx Val Acc', current_max)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        tf.keras.backend.clear_session()
        gc.collect()

    return model, current_max

# def ClassBalancedProbabilisticCurriculumTrain(
#     model,
#     x,
#     y,
#     data_sizes,
#     batch_size=32,
#     epochs=1,
#     callbacks=None,
#     verbose=2,
#     ct_type="easy",
#     val_data = None
# ):
#     current_max = 0
#     for i in range(epochs):

#         scores = get_ind_loss(model, x, y)

#         if ct_type == "easy":
#             scores = 1 / scores
#         elif ct_type == "hard":
#             scores = scores
#         elif ct_type == "random":
#             scores = np.ones(len(y))


#         index = get_class_balanced_index_by_scores(data_sizes[i], y, scores)
        
#         sub_x = x[index]
#         sub_y = y[index]

#         history = model.fit(
#               sub_x,
#               sub_y,
#               epochs=1,
#               batch_size=batch_size,
#               verbose=verbose,
#               callbacks=callbacks,
#               validation_data = val_data
#           )
        
#         current = history.history["val_accuracy"][0]
#         current_max = max(current_max, current)
#         # "{:.4f}".format(max(self.history["acc"]))
#         # print('Current Mx Val Acc', current_max)
#         print("Current Max Val Acc", "{:.4f}".format(current_max))
#         tf.keras.backend.clear_session()
#         gc.collect()

#     return model

def ClassBalancedProbabilisticCurriculumTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    val_data = None,
    scores = None
):
    current_max = 0
    for i in range(epochs):

        index = get_class_balanced_index_by_scores(data_sizes[i], y, scores)
        
        sub_x = x[index]
        sub_y = y[index]

        history = model.fit(
              sub_x,
              sub_y,
              epochs=1,
              batch_size=batch_size,
              verbose=verbose,
              callbacks=callbacks,
              validation_data = val_data
          )
        
        current = history.history["val_accuracy"][0]
        current_max = max(current_max, current)
        # "{:.4f}".format(max(self.history["acc"]))
        # print('Current Mx Val Acc', current_max)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        tf.keras.backend.clear_session()
        gc.collect()

    return model, current_max

def ClassBalancedMixupPrevCurriculumTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    ct_type="easy",
    alpha=0.4,
    val_data=None,
    scores=None,
):
    current_max = 0
    for i in range(epochs):

        if scores is not None:
            mixupX, mixupY = get_class_balanced_mixup_by_prev_scores(
                x, y, scores, data_sizes[i], ct_type=ct_type, alpha=alpha
            )
        else:
            mixupX, mixupY = get_class_balanced_mixup_by_model_prev_scores(
                model, x, y, data_sizes[i], ct_type=ct_type, alpha=alpha
            )

        history = model.fit(
            mixupX,
            mixupY,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_data,
        )
        current = history.history["val_accuracy"][0]
        current_max = max(current_max, current)
        # "{:.4f}".format(max(self.history["acc"]))
        # print('Current Mx Val Acc', current_max)
        print("Current Max Val Acc", "{:.4f}".format(current_max))

        tf.keras.backend.clear_session()
        gc.collect()

    return model

def MixupPrevCurriculumTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    ct_type = "easy",
    alpha=0.4,
    val_data=None,
    scores=None,
):
    current_max = 0
    for i in range(epochs):

        if scores is not None:
            mixupX, mixupY = get_mixup_by_prev_scores(
                x, y, scores, data_sizes[i], ct_type=ct_type, alpha=alpha
            )
        else:
            mixupX, mixupY = get_mixup_by_model_prev_scores(
                model, x, y, data_sizes[i], ct_type=ct_type, alpha=alpha
            )

        history = model.fit(
            mixupX,
            mixupY,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_data,
        )
        current = history.history["val_accuracy"][0]
        current_max = max(current_max, current)
        # "{:.4f}".format(max(self.history["acc"]))
        # print('Current Mx Val Acc', current_max)
        print("Current Max Val Acc", "{:.4f}".format(current_max))

        tf.keras.backend.clear_session()
        gc.collect()

    return model, current_max


def MixupAfterCurriculumTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    anti=False,
    alpha=0.4,
):
    for i in range(epochs):

        mixupX, mixupY = get_mixup_by_model_after_scores(
            model, x, y, data_sizes[i], anti=anti, alpha=alpha
        )

        model.fit(
            mixupX,
            mixupY,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        tf.keras.backend.clear_session()
        gc.collect()

    return model


def mixupBothLossCurriculumTrain(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    anti=False,
    alpha=0.4,
):
    for i in range(epochs):

        mixupX, mixupY = get_both_losses(
            model, x, y, data_sizes[i], anti=anti, alpha=alpha
        )

        model.fit(
            mixupX,
            mixupY,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        tf.keras.backend.clear_session()
        gc.collect()

    return model





def CyclicalTrainSLP3(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    score_type="",
    reverse=False,
):

    individual_loss_cal = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.compat.v1.losses.Reduction.NONE
    )

    total_sample_count = len(x)
    ind_val_loss = np.zeros(total_sample_count)
    true_label = np.argmax(y, axis=-1)
    label_pred = np.zeros(total_sample_count)
    Mov_label_change = np.zeros(total_sample_count)  # 8
    GAMA = 0.9
    index = None
    sample_select_count = np.zeros(total_sample_count)

    for i in range(epochs):
        pred = model.predict(x, 1024)
        ind_loss = individual_loss_cal(y, pred).numpy()  # 1
        softmax_scores = get_model_softmax_scores(x, y, model)  # 2
        label_pred = np.argmax(pred, axis=-1)
        true_preds = np.where(true_label == label_pred, 1.0, 0.0)

        if i == 0:
            MovLoss_i = ind_loss.copy()  # 3
            MovLoss = ind_loss.copy()  # 4
            Mov_dLoss_i = ind_loss.copy()  # 5
            Mov_dLoss = ind_loss.copy()  # 6
            Mov_True_Pred = true_preds.copy() + 0.01
            ind_loss_dif = np.ones(total_sample_count)  # 7

        else:

            prev_ind_loss = ind_val_loss.copy()
            prev_label_pred = label_pred.copy()
            ind_loss_dif = abs(prev_ind_loss - ind_loss)
            label_change = np.where(prev_label_pred == label_pred, 0, 1)

            MovLoss_i[index] = (GAMA * ind_val_loss[index]) + (
                (1 - GAMA) * MovLoss_i[index]
            )
            MovLoss = (GAMA * ind_val_loss) + ((1 - GAMA) * MovLoss)

            Mov_dLoss_i[index] = (GAMA * ind_loss_dif[index]) + (
                (1 - GAMA) * Mov_dLoss_i[index]
            )
            Mov_dLoss = (GAMA * ind_loss_dif) + ((1 - GAMA) * Mov_dLoss)

            Mov_True_Pred = (GAMA * true_preds) + ((1 - GAMA) * Mov_True_Pred)
            Mov_label_change = (GAMA * label_change) + ((1 - GAMA) * Mov_label_change)

            sample_select_count[index] += 1

        if score_type == "loss":
            scores = ind_loss
        elif score_type == "softmax":
            scores = softmax_scores
        elif score_type == "ind_loss_dif":
            scores = ind_loss_dif
        elif score_type == "MovLoss":
            scores = MovLoss
        elif score_type == "MovLoss_i":
            scores = MovLoss_i
        elif score_type == "Mov_dLoss":
            scores = Mov_dLoss
        elif score_type == "Mov_dLoss_i":
            scores = Mov_dLoss_i
        elif score_type == "Mov_True_Pred":
            scores = Mov_True_Pred
        elif score_type == "random":
            scores = np.ones(total_sample_count)
        elif score_type == "loss_mov_d_loss":
            scores = (1 / (ind_loss + 0.01)) * (1 / (Mov_dLoss + 0.01))
        elif score_type == "loss_mov_true_pred":
            scores = (1 / (ind_loss + 0.01)) * Mov_True_Pred
        elif score_type == "loss_softmax":
            scores = (1 / (ind_loss + 0.01)) * softmax_scores
        elif score_type == "mov_d_loss_mov_true_pred":
            scores = (1 / (Mov_dLoss + 0.01)) * Mov_True_Pred
        elif score_type == "mov_d_loss_softmax":
            scores = (1 / (Mov_dLoss + 0.01)) * softmax_scores
        elif score_type == "mov_true_pred_softmax":
            scores = Mov_True_Pred * softmax_scores
        elif score_type == "loss_mov_d_loss2":
            scores = (1 / (ind_loss + 0.01)) + (1 / (Mov_dLoss + 0.01))
        elif score_type == "loss_mov_true_pred2":
            scores = (1 / (ind_loss + 0.01)) + Mov_True_Pred
        elif score_type == "loss_softmax2":
            scores = (1 / (ind_loss + 0.01)) + softmax_scores
        elif score_type == "mov_d_loss_mov_true_pred2":
            scores = (1 / (Mov_dLoss + 0.01)) + Mov_True_Pred
        elif score_type == "mov_d_loss_softmax2":
            scores = (1 / (Mov_dLoss + 0.01)) + softmax_scores
        elif score_type == "mov_true_pred_softmax2":
            scores = Mov_True_Pred * softmax_scores
        else:
            print("score type uygun değil")
            1 / 0
        if reverse:
            scores = 1 / (scores + 0.01)

        sample_count_epoch = int(total_sample_count * data_sizes[i])
        index = np.random.choice(
            range(0, len(x)), sample_count_epoch, p=scores / scores.sum(), replace=False
        )

        sub_x, sub_y = x[index], y[index]

        model.fit(
            sub_x,
            sub_y,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        tf.keras.backend.clear_session()
        gc.collect()

    return model


def CyclicalTrainSLP2(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
    score_type="",
):
    sample_count = len(x)
    DIH = np.ones(sample_count)
    dDIH = np.ones(sample_count)
    ind_val_loss = np.ones(sample_count)
    true_label = np.argmax(y, axis=-1)
    label_pred = np.ones(sample_count)
    true_preds = np.ones(sample_count)
    label_change = np.ones(sample_count)
    individual_loss_cal = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.compat.v1.losses.Reduction.NONE
    )

    GAMA = 0.9
    index_update = True
    for i in range(epochs):

        prev_ind_loss = ind_val_loss.copy()
        prev_label_pred = label_pred.copy()

        softmax_scores = get_model_softmax_scores(x, y, model)
        dLoss = abs(prev_ind_loss - ind_val_loss)

        if score_type == "loss":
            scores = ind_val_loss
        elif score_type == "dih":
            scores = DIH
        elif score_type == "dLoss":
            scores = dLoss
        elif score_type == "dDIH":
            scores = dDIH
        elif score_type == "true_preds":
            scores = true_preds
        elif score_type == "label_change":
            scores = label_change
        elif score_type == "softmax_scores":
            scores = softmax_scores

        index = np.random.choice(
            range(0, len(x)), data_sizes[i], p=scores / scores.sum(), replace=False
        )

        model.fit(
            x[index],
            y[index],
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        pred = model.predict(x, 1024)

        ind_val_loss = individual_loss_cal(y, pred).numpy()

        if index_update:

            DIH[index] = (GAMA * ind_val_loss[index]) + ((1 - GAMA) * DIH[index])

            dDIH[index] = (GAMA * dLoss[index]) + ((1 - GAMA) * dDIH[index])

            label_pred = np.argmax(pred, axis=-1)

            true_preds[index] = GAMA * (
                np.where(true_label == label_pred, 1.0, 0.0)[index]
            ) + ((1 - GAMA) * true_preds[index])

            label_change = GAMA * (
                np.where(prev_label_pred == label_pred, 0, 1)[index]
            ) + ((1 - GAMA) * label_change[index])

        tf.keras.backend.clear_session()
        gc.collect()

    return model


def ConditionTrain(
    model,
    x,
    y,
    condition,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=2,
):
    for i in range(epochs):

        scores = get_model_softmax_scores(x, y, model)

        c = eval(condition)

        index = np.where(c)[0]

        sub_x = x[index]
        sub_y = y[index]

        model.fit(
            sub_x,
            sub_y,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        tf.keras.backend.clear_session()
        gc.collect()

    return model


def CurriculumTrain(model, x, y, epochs, batch_size=32, callbacks=None, verbose=1, validation_data = None):
    current_max = 0
    
    for i, e in enumerate(epochs):
        x_train = _get_concat_x(x, 0, i+1)
        y_train = _get_concat_x(y, 0, i+1)
        history = model.fit(
            x_train,
            y_train,
            epochs=e,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data = validation_data
        )
        current = max(history.history["val_accuracy"])
        current_max = max(current_max, current)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        tf.keras.backend.clear_session()
        gc.collect()
    return model, current_max

def CurriculumTrain2(model, x, y, epochs, batch_size=32, callbacks=None, verbose=1, validation_data = None):
    current_max = 0
    
    for i, e in enumerate(epochs):
        x_train = x[i]
        y_train = y[i]
        history = model.fit(
            x_train,
            y_train,
            epochs=e,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data = validation_data
        )
        current = max(history.history["val_accuracy"])
        current_max = max(current_max, current)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        tf.keras.backend.clear_session()
        gc.collect()
    return model, current_max

def SPLTrain(model, x, y, epochs, batch_size=32, callbacks=None, verbose=2, validation_data = None,
             init_percent = 1.0, th = 0.2, K = 0.75, anti = False):
    
    # print('SPL Train')
    
    current_max = 0

    index = np.random.choice(
        range(0, len(y)), int(len(y) * init_percent), p=None, replace=False
    )
        
    if isinstance(x, list):
        x_train = []
        for i in x:
            x_train.append(i[index])
    else:
        x_train = x[index]
    if isinstance(y, list):
        y_train = []
        for i in x:
            y_train.append(i[index])
    else:
        y_train = y[index]  
    
    h = model.fit(x_train, y_train, batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_data = validation_data)
    tf.keras.backend.clear_session()
    gc.collect()
    th = h.history['loss'][0]
    
    # h = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=verbose)
    
    # individual_loss_cal = tf.keras.losses.CategoricalCrossentropy()
    
    # pred = model.predict(x, 1024)
    # th = individual_loss_cal(y, pred).numpy()
    
    total_train = init_percent
    
    # print('2')
    # print('--------------------')
    # print(total_train)
    # print(epochs)
    
    while total_train < epochs - 1:
        # print('3')
        scores = get_ind_loss(model, x, y, batch_size = batch_size)
        # if len(y.shape) > 2:
        #     scores = np.sum(scores,axis = 1)
        #     th = np.mean(scores)
        tf.keras.backend.clear_session()
        gc.collect()  
        # print('4')
        if anti == False: 
            index = np.where(scores < th)[0]
        if anti == True:
            index = np.where(scores > th)[0]
        np.random.shuffle(index)
        # print(len(index))
        # print(index)
        # print('5')
        # print("Len İndex", len(index), index[:5])

        tf.keras.backend.clear_session()
        gc.collect()        

        if isinstance(x, list):
            x_train = []
            for i in x:
                x_train.append(i[index])
        else:
            x_train = x[index]
        
        if isinstance(y, list):
            y_train = []
            for i in x:
                y_train.append(i[index])
        else:
            y_train = y[index]        
        
        # print('6')
        
        # y_train = y[index]
        history = model.fit(
            x_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data = validation_data
        )
        
        # print('7')
        
        total_train += len(y_train) / len(y)
        if anti == False:
            th *= 1 / K
        if anti == True:
            th *= K

        
        # print('% Percent:', len(y_train) / len(y), "Threshold: ", th, "Len İndex: ", len(index))
        
        current = max(history.history["val_accuracy"])
        current_max = max(current_max, current)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        tf.keras.backend.clear_session()
        gc.collect()
        
        # print("*************************")
        # print(scores)
        # print(y)
        # print("*************************")  
        
    history = model.fit(
        x,
        y,
        epochs=1,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        validation_data = validation_data
    )
    current = max(history.history["val_accuracy"])
    current_max = max(current_max, current)
    print("Current Max Val Acc", "{:.4f}".format(current_max))
    tf.keras.backend.clear_session()
    gc.collect()


    
    return model, current_max

from split_data import get_class_balanced_indies, split_data_x_y, load_scores, split_data_x

from ChangeDatasize import get_model_softmax_scores, get_ind_loss, get_sparse_ind_loss, generic_shuffle_data



def StageSplTrain(model, x, y, epochs, N, weights, batch_size=32, callbacks=None, verbose=2, validation_data = None, anti = False):
    
    # scores = get_ind_loss(model, x, y, batch_size = batch_size)
    # tf.keras.backend.clear_session()
    # gc.collect()
    # if not anti:
    #     scores = 1 / scores

    
    scores = np.random.rand(_get_range_size(x, y))

    sorted_index = get_class_balanced_indies(scores, y, y.shape[-1], N, weights)
    
    s_x, s_y = split_data_x_y(x, y, N, arr_ind=sorted_index, weights=weights)

    current_max = 0

    for i, e in enumerate(epochs):

        for _ in range(e):
            x_train = _get_concat_x(s_x, 0, i+1)
            y_train = _get_concat_x(s_y, 0, i+1)
            
            x_train, y_train = generic_shuffle_data(x_train, y_train)        
            history = model.fit(
                x_train,
                y_train,
                epochs=1,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks,
                validation_data = validation_data
            )
            current = max(history.history["val_accuracy"])
            current_max = max(current_max, current)
            print("Current Max Val Acc", "{:.4f}".format(current_max))
            tf.keras.backend.clear_session()
            gc.collect()
            scores = get_ind_loss(model, x, y, batch_size = batch_size)
            if not anti:
                scores = 1 / scores
            sorted_index = get_class_balanced_indies(scores, y, y.shape[-1], N, weights)
            
            s_x, s_y = split_data_x_y(x, y, N, arr_ind=sorted_index, weights=weights)
    
            tf.keras.backend.clear_session()
            gc.collect()
    return model, current_max    








