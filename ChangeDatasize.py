import numpy as np
import pickle


def get_train_data_by_scores(x, y, size, scores=None):
    index = np.random.choice(range(0, len(x)), size, p=scores, replace=False)
    Train_x = x[index]
    Test_y = y[index]
    return Train_x, Test_y


def get_cycle_data_sizes(
    size, start_percent, multiplier, end_percent, strategy, start_increase, total_sample
):
    sizes = []
    current_size = int(size * start_percent)
    sizes.append(current_size)
    max_size = int(size * max(end_percent, start_percent))
    min_size = int(size * min(end_percent, start_percent))

    if strategy not in ["cycle", "lower_to_bigger", "bigger_to_lower"]:
        raise Exception(
            "strategy must be one of these ['cycle','lower_to_bigger','bigger_to_lower']"
        )

    if strategy == "cycle":
        while sum(sizes) < total_sample:
            if start_increase:
                current_size = int(current_size * (1 / multiplier))
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

    return sizes


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


def get_model_softmax_scores(x, y, model):
    if "sklearn" in str(type(model)):
        model.fit(x, np.argmax(y, axis=-1))
        p = model.predict_proba(x)
    elif "keras" in str(type(model)):
        p = model.predict(x)

    real_ind = np.argmax(y, axis=1)
    scores = p[np.arange(len(p)), real_ind]

    return scores


def load_scores(path):
    scores = pickle.load(open(path, "rb"))
    return scores
