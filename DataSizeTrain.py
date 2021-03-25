import gc
import tensorflow as tf
from ChangeDatasize import get_train_data_by_scores


def ChangingDatasizeTrain(
    model,
    x,
    y,
    data_sizes,
    scores=None,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=1,
):

    for i in range(epochs):
        sub_x, sub_y = get_train_data_by_scores(
            x, y, data_sizes[i], scores=scores / scores.sum()
        )

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

from ChangeDatasize import get_model_softmax_scores

def ChangingDatasizeTrainSLP(
    model,
    x,
    y,
    data_sizes,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=1,
):
    
    
    
    for i in range(epochs):
        
        scores = get_model_softmax_scores(x, y, model)
        
        sub_x, sub_y = get_train_data_by_scores(
            x, y, data_sizes[i], scores=scores / scores.sum()
        )

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