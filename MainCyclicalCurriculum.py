import numpy as np

import argparse
import pickle
import random
import os
import tensorflow as tf

import datasets
import models

import gc

from ChangeDatasize import get_cycle_data_sizes, load_scores, get_cycle_sizes

from loss_history import (
    BatchHistory,
    WeightsEpochHistory,
    DynamicLossEpochHistory,
    EpochHistory,
    WeightsBatchHistory,
    Without_O_Accuracy,
)

from Trains import CyclicalTrain, AgingScoredCyclicalTrain, CyclicalTrainSLP3, CyclicalTrainSLP

# from CyclicalLearningRate import CyclicLR

def write_file(dataset, curriculum_type, seed, current_max):
    name = (
        str(dataset)
        + "_"
        + str(curriculum_type)
        + "_s_"
        + str(seed)
    )
    
    f = open("Skorlar.txt", "a")
    
    f.write(str(name) + "\t" + str("{:.4f}".format(current_max)) + '\n')
    
    f.close()    

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
parser.add_argument("-ct", "--curriculum_type", default="easy", type=str)

parser.add_argument("--min_lr", default=0.01, type=float)
parser.add_argument("--max_lr", default=0.01, type=float)
parser.add_argument("--lr_multiplier", default=1.0, type=float)
parser.add_argument("--lr_decay", default=1.0, type=float)

parser.add_argument("-spath", "--score_path", default="", type=str)
parser.add_argument("-spl_st", "--spl_score_type", default="", type=str)
parser.add_argument("-st_reverse", "--spl_score_type_reverse", default=0, type=int)

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
score_type = results.spl_score_type
st_reverse = results.spl_score_type_reverse






os.environ["PYTHONHASHSEED"] = str(seed)

random.seed(seed)

np.random.seed(seed)

tf.random.set_seed(seed)

# os.environ["TF_DETERMINISTIC_OPS"] = "1"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dataset_names = ['20_news','sarcasm','reuters','hotel','stweet','ctweet','qpair','food',
                    'sof','toxic','reddit','ner','turkish_ner','ttc4900','30columnists','2500kose',
                    'cifar10', 'cifar100', 'fashion', 'stl_10']

# dataset_names = ['reuters']

# curriculum_names = ['vanilla','easy','random','hard','StageSPL','Anti-StageSPL']

# curriculum_names = ['cycle']

# curriculum_names = ['cycle','splcycle','111cycle','111splcycle','vanilla']

curriculum_names = ['cycle']

batch_sizes = [128, 128, 512, 100, 1024, 32, 100, 64, 32, 128, 128, 32, 32, 64, 64, 64, 128, 128, 128, 32]
epochs = [20, 10, 20, 8, 6, 6, 6, 6, 6, 4, 6, 12, 12, 16, 30, 30, 30, 48, 39, 24]
seed_values = [0, 42, 64, 120, 1234]
xepochs = [1, 1, 1, 1, 1]
curriculum_types = ['easy','random','hard']



# score_path_names = [20_news]

directory = "res/"

import os
if not os.path.exists(directory):
    os.makedirs(directory)

iiiii = 0

for d1,d2,d3 in zip(dataset_names[16:20],batch_sizes[16:20],epochs[16:20]):
    for c1 in curriculum_names[0:1]:
        for lamda in [None, 5, 0.5, 0.25][0:1]:
            for seed, x in zip(seed_values[:],xepochs[:]):   
                if iiiii < 0:
                    pass
                else:
                    dataset = d1
                    # curriculum_type = c1
                    batch_size = d2
                    vanilla_epoch = d3 * x
                    epochs = d3 * x
                    # epochs = []
                    # epochs.append(d3 // 2)
                    # epochs.append(d3 // 2)
                    # epochs.append(d3 // 2)
                    
                    # curriculum_type = "hard"
                    
                    score_path = dataset + "_scores.p"
                    
                    os.environ["PYTHONHASHSEED"] = str(seed)
            
                    random.seed(seed)
            
                    np.random.seed(seed)
            
                    tf.random.set_seed(seed)
                    
        
                    X_train, y_train, X_test, y_test = datasets.get_data(dataset)
                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                    if dataset == 'bert_imdb1' or True:
                        try:
                            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                            tf.config.experimental_connect_to_cluster(tpu)
                            tf.tpu.experimental.initialize_tpu_system(tpu)
                            strategy = tf.distribute.experimental.TPUStrategy(tpu)
                        
                            # Create model
                            with strategy.scope():
                                model = models.get_model(dataset)
                        except:
                            model = models.get_model(dataset)
                    else:
                        model = models.get_model(dataset)
                    model.build(np.array(X_train).shape)
                    weights = model.get_weights()
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    val_data = (X_test, y_test)
                    
                    weights_history = WeightsEpochHistory()
                    EpochHistory1 = EpochHistory(data = (X_train, y_train), batch_size=batch_size)
                    # EpochHistory2 = EpochHistory(data = (X_test, y_test), batch_size=batch_size)
                    
                    n = round((np.ceil(len(X_train) / batch_size) * epochs) / 300)
                    
                    BatchHistory1 = BatchHistory(data = (X_test, y_test), batch_size=batch_size, n = n)
                    
        
                    callbacks = []
                    if dataset == 'ner' or dataset == 'turkish_ner':
                        callbacks.append(Without_O_Accuracy((X_test, y_test)))
                    
                    # callbacks.append(weights_history)
                    # callbacks.append(EpochHistory1)
                    # callbacks.append(BatchHistory1)
                    
                    # if curriculum_type == "random":
                    #     scores = np.ones(len(X_train))
                    # else:
                    #     scores = load_scores(score_path)
                    #     if curriculum_type == "easy":
                    #         scores = 1 / scores
                    #     elif curriculum_type == "hard":
                    #         scores = scores
                    #     elif curriculum_type == "mediocre":
                    #         scores = 1 - abs(scores - 0.5)
                    #     elif curriculum_type == "reverse_mediocre":
                    #         scores = abs(scores - 0.5)
                    scores = load_scores(score_path)
                    
        
                    
                    # easy_scores = 1 / scores
                    
                    # lamda = 1.5
                    
                    if lamda is not None: 
                        easy_scores = lamda * np.exp((-1 * lamda) * scores)
                        min_gt0 = np.ma.array(easy_scores, mask=easy_scores<=0).min(0)
                        easy_scores = np.where(easy_scores == 0, min_gt0, easy_scores)
                    else:
                        easy_scores = 1 / scores
                    
                    
                    hard_scores = scores
                    rand_scores = np.ones(len(X_train))
                    
                    
                    
                    # easy_scores += np.std(easy_scores)
        
                    
                    train_data_size = len(X_train)
                    total_sample = train_data_size * epochs
                    
                    cycle_step = np.ceil(train_data_size / batch_size) * 2
                    
                    n = np.ceil(np.ceil(train_data_size / batch_size) * epochs / sample_count)
                    
                    data_sizes = ([1] * (epochs // 2))
                    
                    tmp_data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs - epochs // 2)
                    
                    # data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs)
                    
                    data_sizes.extend(tmp_data_sizes)
    
                    # c1 = 'cycle'
                    # curriculum_type = 'easy'
                    # scores = easy_scores
                    if c1 == 'cycle':
                        
                        class CustomSaver(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs={}):
                                    self.model.save("model_{}.hd5".format(epoch))
                        
                        # start_percent = 0.25
                        # end_percent = 1
                        # multiplier = 0.25
                        data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs)
                        # data_sizes = [0.25, 0.5, 1, 1, 1, 0.5, 0.25, 0.5, 1, 1, 1, 0.5, 0.25, 0.5, 1, 1, 1,
                        #               0.5, 0.25, 0.5, 1, 1, 1, 0.5, 0.25, 0.5, 1, 1, 1, 0.5, 0.25, 0.5, 1, 1]
                        curriculum_type = 'easy_cycle' + '_lamda_'+str(lamda)
                        scores = easy_scores
                        model, current_max, results = CyclicalTrain(
                            model,
                            X_train,
                            y_train,
                            data_sizes,
                            scores=scores,
                            batch_size=batch_size,
                            epochs=len(data_sizes),
                            callbacks=callbacks,
                            data=(X_test, y_test),
                        )
                        
                        # model, current_max, results = CyclicalTrainSLP(
                        #     model,
                        #     X_train,
                        #     y_train,
                        #     data_sizes,
                        #     batch_size=batch_size,
                        #     epochs=len(data_sizes),
                        #     callbacks=callbacks,
                        #     data=(X_test, y_test),
                        # )
                        
                        write_file(dataset, curriculum_type, seed, current_max)
                        # pickle.dump(results, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(weights_history.weights, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_weights_" + str(seed) + ".p","wb"))
                        # pickle.dump(EpochHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(BatchHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                    
                    if c1 == 'splcycle':
                        
                        epochs += 1
                        
                        
                        data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs)
                        curriculum_type = 'easy_splcycle' + '_lamda_'+str(lamda)
                        scores = easy_scores
                        
                        
                        data_sizes = data_sizes[2:]
                        
                        
                        model, current_max, results = CyclicalTrainSLP(
                            model,
                            X_train,
                            y_train,
                            data_sizes,
                            batch_size=batch_size,
                            epochs=len(data_sizes),
                            callbacks=callbacks,
                            data=(X_test, y_test),
                        )
                        
                        write_file(dataset, curriculum_type, seed, current_max)
                        # pickle.dump(results, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(weights_history.weights, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_weights_" + str(seed) + ".p","wb"))
                        # pickle.dump(EpochHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(BatchHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                    
                    if c1 == '111cycle':
                        # data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs)
                                        
                        data_sizes = ([1] * (epochs // 2))
                        
                        tmp_data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs - epochs // 2)
                        
                        data_sizes.extend(tmp_data_sizes)
                    
                        curriculum_type = 'easy_111cycle' + '_lamda_'+str(lamda)
                        scores = easy_scores
    
                        model, current_max, results = CyclicalTrainSLP(
                            model,
                            X_train,
                            y_train,
                            data_sizes,
                            batch_size=batch_size,
                            epochs=len(data_sizes),
                            callbacks=callbacks,
                            data=(X_test, y_test),
                        )
                        
                        write_file(dataset, curriculum_type, seed, current_max)
                        # pickle.dump(results, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(weights_history.weights, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_weights_" + str(seed) + ".p","wb"))
                        # pickle.dump(EpochHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(BatchHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                        
                    if c1 == '111splcycle':
                        # data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs)
                        
                        data_sizes = ([1] * (epochs // 2))
                        
                        tmp_data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, epochs - epochs // 2)
                        
                        data_sizes.extend(tmp_data_sizes)
                        
                        curriculum_type = 'easy_111splcycle' + '_lamda_'+str(lamda)
                        scores = easy_scores
    
                        model, current_max, results = CyclicalTrainSLP(
                            model,
                            X_train,
                            y_train,
                            data_sizes,
                            batch_size=batch_size,
                            epochs=len(data_sizes),
                            callbacks=callbacks,
                            data=(X_test, y_test),
                        )
                        
                        write_file(dataset, curriculum_type, seed, current_max)
                        # pickle.dump(results, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(weights_history.weights, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_weights_" + str(seed) + ".p","wb"))
                        # pickle.dump(EpochHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                        # pickle.dump(BatchHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                    
                    
                    # n = round((np.ceil(len(X_train) / batch_size) * epochs) / 300)
                    
                    # BatchHistory1 = BatchHistory(data = (X_test, y_test), batch_size=batch_size, n = n)
                    
        
                    # callbacks = []
                    # if dataset == 'ner' or dataset == 'turkish_ner':
                    #     callbacks.append(Without_O_Accuracy((X_test, y_test)))
                    
                    # # callbacks.append(weights_history)
                    # # callbacks.append(EpochHistory1)
                    # callbacks.append(BatchHistory1)
                    
                    
                    # c1 = 'cycle'
                    # model = models.get_model(dataset)
                    # model.build(np.array(X_train).shape)
                    # model.set_weights(weights)
                    # curriculum_type = 'hard'
                    # scores = hard_scores
                    # if c1 == 'cycle':
                    
                    #     model, current_max, results = CyclicalTrain(
                    #         model,
                    #         X_train,
                    #         y_train,
                    #         data_sizes,
                    #         scores=scores,
                    #         batch_size=batch_size,
                    #         epochs=len(data_sizes),
                    #         callbacks=callbacks,
                    #         data=(X_test, y_test),
                    #     )
                    #     write_file(dataset, curriculum_type, seed, current_max)
                    #     # pickle.dump(results, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_train_results_" + str(seed) + ".p","wb"))
                    #     # pickle.dump(weights_history.weights, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_weights_" + str(seed) + ".p","wb"))
                    #     # pickle.dump(EpochHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))            
                    #     pickle.dump(BatchHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                        
                    # n = round((np.ceil(len(X_train) / batch_size) * epochs) / 300)
                    
                    # BatchHistory1 = BatchHistory(data = (X_test, y_test), batch_size=batch_size, n = n)
                    
        
                    # callbacks = []
                    # if dataset == 'ner' or dataset == 'turkish_ner':
                    #     callbacks.append(Without_O_Accuracy((X_test, y_test)))
                    
                    # # callbacks.append(weights_history)
                    # # callbacks.append(EpochHistory1)
                    # callbacks.append(BatchHistory1)
                    
                    
                    # c1 = 'cycle'
                    # model = models.get_model(dataset)
                    # model.build(np.array(X_train).shape)
                    # model.set_weights(weights)
                    # curriculum_type = 'rand'
                    # scores = hard_scores
                    # if c1 == 'cycle':
                    
                    #     model, current_max, results = CyclicalTrain(
                    #         model,
                    #         X_train,
                    #         y_train,
                    #         data_sizes,
                    #         scores=scores,
                    #         batch_size=batch_size,
                    #         epochs=len(data_sizes),
                    #         callbacks=callbacks,
                    #         data=(X_test, y_test),
                    #     )
                    #     write_file(dataset, curriculum_type, seed, current_max)
                    #     # pickle.dump(results, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_train_results_" + str(seed) + ".p","wb"))
                    #     # pickle.dump(weights_history.weights, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_weights_" + str(seed) + ".p","wb"))
                    #     # pickle.dump(EpochHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))            
                    #     pickle.dump(BatchHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                        
                    # n = round((np.ceil(len(X_train) / batch_size) * epochs) / 300)
                    
                    # BatchHistory1 = BatchHistory(data = (X_test, y_test), batch_size=batch_size, n = n)
                    
        
                    # callbacks = []
                    # if dataset == 'ner' or dataset == 'turkish_ner':
                    #     callbacks.append(Without_O_Accuracy((X_test, y_test)))
                    
                    # # callbacks.append(weights_history)
                    # # callbacks.append(EpochHistory1)
                    # callbacks.append(BatchHistory1)
                    
                    
                    # c1 = 'vanilla'
                    # model = models.get_model(dataset)
                    # model.build(X_train.shape)
                    # model.set_weights(weights)
                    # curriculum_type = ''
                    
                    if c1 == 'vanilla':
                        tf.keras.backend.clear_session()
                        gc.collect()
                        history = model.fit(X_train, y_train, batch_size=batch_size,
                                            epochs=vanilla_epoch, validation_data = val_data,
                                            callbacks = callbacks)
                        results = history.history
                        current_max = 0
                        current = max(history.history["val_accuracy"])
                        current_max = max(current_max, current)
                        tf.keras.backend.clear_session()
                        gc.collect()
                        write_file(dataset, curriculum_type, seed, current_max)
                    #     # pickle.dump(results, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_train_results_" + str(seed) + ".p","wb"))
                    #     # pickle.dump(weights_history.weights, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_weights_" + str(seed) + ".p","wb"))
                    #     # pickle.dump(EpochHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                    #     pickle.dump(BatchHistory1.history, open(directory + c1 + "_" + curriculum_type + '_' + dataset + "_full_train_results_" + str(seed) + ".p","wb"))
                iiiii += 1
                        
                        
                        
                        
                        
                        
                    
                
            
            
            
            
            
            
            
            
            
            