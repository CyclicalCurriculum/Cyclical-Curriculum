import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
path = "/content/drive/My Drive/Datasets/"

def get_data(name):
    if name == "cifar10":
        return get_cifar10_data()
    elif name == "cifar100":
        return get_cifar100_data()
    elif name == "fashion":
        return get_fashion_data()
    elif name == "mnist":
        return get_mnist_data()
    elif name == "kmnist":
        return get_kmnist_data()
    elif name == "beans":
        return get_beans_data()
    elif name == "food101":
        return get_food101_data()
    elif name == "tf_flowers":
        return get_tf_flowers_data()
    elif name == "oxford_flowers":
        return get_oxford_flowers_data()
    elif name == "aircraft":
        return get_aircraft_data()
    elif name == "svhn":
        return get_svhn_data()
    elif name == "stl_10":
        return get_stl_10_data()
    elif name == "imdb":
        return get_imdb_data()
    elif name == "imdb1":
        return get_imdb1_data()
    elif name == "sarcasm":
        return get_sarcasm_data()
    elif name == "hotel":
        return get_hotel_data()
    elif name == 'stweet':
        return get_stweet_data()
    elif name == 'qpair':
        return get_qpair_data()
    elif name == 'ctweet':
        return get_ctweet_data()
    elif name == 'food':
        return get_food_data()
    elif name == 'sof':
        return get_sof_data()
    elif name == 'reddit':
        return get_reddit_data()
    elif name == "toxic":
        return get_toxic_data()
    elif name == 'reuters':
        return get_reuters_data()
    elif name == "20_news":
        return get_20_news_data()
    elif name == 'squad':
        return get_squad_data()
    elif name == 'bert_imdb':
        return get_bert_imdb_data()
    elif name == 'ner':
        return get_ner_data()
    elif name == 'turkish_ner':
        return get_turkish_ner_data()
    elif name == 'ttc4900':
        return get_ttc4900_data()
    elif name == '30columnists':
        return get_30columnists_data()
    elif name == '2500kose':
        return get_2500kose_data()
    else:
        return get_ucl_data(name)


def get_cifar10_data():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.cifar10.load_data()
    train_images, test_images = (
        train_images.astype("float32") / 255.0,
        test_images.astype("float32") / 255.0,
    )
    train_images = train_images.reshape((train_images.shape[0], 32, 32, 3))
    test_images = test_images.reshape((test_images.shape[0], 32, 32, 3))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def get_cifar100_data():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.cifar100.load_data()
    train_images, test_images = (
        train_images.astype("float32") / 255.0,
        test_images.astype("float32") / 255.0,
    )
    train_images = train_images.reshape((train_images.shape[0], 32, 32, 3))
    test_images = test_images.reshape((test_images.shape[0], 32, 32, 3))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def get_fashion_data():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.fashion_mnist.load_data()
    train_images, test_images = (
        train_images.astype("float32") / 255.0,
        test_images.astype("float32") / 255.0,
    )
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def get_mnist_data():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.mnist.load_data()
    train_images, test_images = (
        train_images.astype("float32") / 255.0,
        test_images.astype("float32") / 255.0,
    )
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def get_kmnist_data():
    (img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'kmnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
    ))

    img_train, img_test = (
        img_train.astype("float32") / 255.0,
        img_test.astype("float32") / 255.0,
    )    

    img_train = img_train.reshape((img_train.shape[0], 28, 28, 1))
    img_test = img_test.reshape((img_test.shape[0], 28, 28, 1))    
    
    label_train = to_categorical(label_train)
    label_test = to_categorical(label_test)    
    
    # print(img_train[0])
    
    return img_train, label_train, img_test, label_test

def get_oxford_flowers_data():

    dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

    image_size = (32, 32)

    num_classes = dataset_info.features["label"].num_classes

    images, labels = [], []
    for img, lab in training_set.as_numpy_iterator():
        images.append(tf.image.resize(img, image_size).numpy())
        labels.append(tf.one_hot(lab, num_classes))
    
    img_train = np.array(images)
    label_train = np.array(labels)

    print(img_train.shape)
    print(img_train[0].shape)
    print(img_train[0])
    print(label_train.shape)
    print(label_train[0].shape)
    print(label_train[0])    

    images, labels = [], []
    for img, lab in test_set.as_numpy_iterator():
        images.append(tf.image.resize(img, image_size).numpy())
        labels.append(tf.one_hot(lab, num_classes))
    
    img_test = np.array(images)
    label_test = np.array(labels)

    img_train, img_test = (
        img_train.astype("float32") / 255.0,
        img_test.astype("float32") / 255.0,
    )    

    img_train = img_train.reshape((img_train.shape[0], 32, 32, 3))
    img_test = img_test.reshape((img_test.shape[0], 32, 32, 3))  


    return img_train, label_train, img_test, label_test

    # tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0


# get_oxford_flowers_data()

def get_svhn_data():
    (img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
    ))

    img_train, img_test = (
        img_train.astype("float32") / 255.0,
        img_test.astype("float32") / 255.0,
    )    

    img_train = img_train.reshape((img_train.shape[0], 32, 32, 3))
    img_test = img_test.reshape((img_test.shape[0], 32, 32, 3))    
    
    label_train = to_categorical(label_train)
    label_test = to_categorical(label_test)    
    
    print(img_train[0])
    
    return img_train, label_train, img_test, label_test



# get_kmnist_data()

def get_beans_data():
    (img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'beans',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
    ))

    label_train = to_categorical(label_train)
    label_test = to_categorical(label_test)    
    
    return img_train, label_train, img_test, label_test

def get_food101_data():
    (img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
    ))
    
    label_train = to_categorical(label_train)
    label_test = to_categorical(label_test)    
    
    print(img_train.shape)
    1 / 0
    
    return img_train, label_train, img_test, label_test

def get_tf_flowers_data():
    
    X_Train, y_train = tfds.as_numpy(tfds.load(
    'tf_flowers',
    split=['train'],
    batch_size=-1,
    as_supervised=True,
    ))[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X_Train, y_train, 
                                                    test_size=0.33, stratify=y_train,
                                                    random_state=42)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test

def get_aircraft_data():
    (img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'malaria',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
    ))

    label_train = to_categorical(label_train)
    label_test = to_categorical(label_test)    
    
    return img_train, label_train, img_test, label_test

def get_imdb1_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_imdb1.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_imdb1.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_imdb1.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_imdb1.pl", "rb"))

    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test    

def get_imdb_data():
    max_features = 5000
    maxlen = 400

    (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(
        num_words=max_features
    )

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test


def get_stl_10_data():
    
    # image_size = (32, 32)
    
    # path = "/content/drive/My Drive/Datasets/"

    train_images = read_all_images(path + "stl_10_train_X.bin")
    print(train_images.shape)

    train_labels = read_labels(path + "stl_10_train_y.bin")
    print(train_labels.shape)

    test_images = read_all_images(path + "stl_10_test_X.bin")
    print(train_images.shape)

    test_labels = read_labels(path + "stl_10_test_y.bin")
    print(train_labels.shape)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # images = []
    # for img in train_images:
    #     images.append(tf.image.resize(img, image_size).numpy())
    
    # train_images = np.array(images)

    # images = []
    # for img in test_images:
    #     images.append(tf.image.resize(img, image_size).numpy())
    
    # test_images = np.array(images)

    return (
        train_images.astype("float32"),
        train_labels,
        test_images.astype("float32"),
        test_labels,
    )


def get_sarcasm_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ""
    X_train = pickle.load(open(path + "X_train_sarcasm.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_sarcasm.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_sarcasm.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_sarcasm.pl", "rb"))
    
    # X_train_sent = pickle.load(open(path + "sarcasm_x_train_sent.pl", "rb"))
    
    

    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test

def get_reuters_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_reuters.p", "rb"))
    X_test = pickle.load(open(path + "X_test_reuters.p", "rb"))
    y_train = pickle.load(open(path + "y_train_reuters.p", "rb"))
    y_test = pickle.load(open(path + "y_test_reuters.p", "rb"))
    
    
    # X_train_sent = pickle.load(open(path + "X_train_reuters_sent.p", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test

def get_hotel_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_hotel.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_hotel.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_hotel.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_hotel.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "X_train_hotel_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test

def get_stweet_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_tweet.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_tweet.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_tweet.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_tweet.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "X_train_tweet_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test    

def get_qpair_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ''
    x_train_question1 = pickle.load(open(path + "x_train_question1.pl", "rb"))
    x_test_question1 = pickle.load(open(path + "x_test_question1.pl", "rb"))
    x_train_question2 = pickle.load(open(path + "x_train_question2.pl", "rb"))
    x_test_question2 = pickle.load(open(path + "x_test_question2.pl", "rb"))    
    X_train = [x_train_question1, x_train_question2]
    X_test = [x_test_question1, x_test_question2]
    
    y_train = pickle.load(open(path + "y_train_question.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_question.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    
    # x_train_question1_sent = pickle.load(open(path + "x_train_question1_sent.pl", "rb"))
    # x_train_question2_sent = pickle.load(open(path + "x_train_question2_sent.pl", "rb"))
    
    
    return X_train, y_train, X_test, y_test 

def get_ctweet_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_ctweet.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_ctweet.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_ctweet.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_ctweet.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    
    # X_train_sent = pickle.load(open(path + "X_train_ctweet_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test 

def get_food_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_food.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_food.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_food.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_food.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "X_train_food_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test     

def get_reddit_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_reddit.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_reddit.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_reddit.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_reddit.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # X_train_sent = pickle.load(open(path + "X_train_reddit_sent.pl", "rb"))

    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test    

def get_toxic_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "Toxic_X_train.pl", "rb"))
    X_test = pickle.load(open(path + "Toxic_X_test.pl", "rb"))
    y_train = pickle.load(open(path + "Toxic_y_train.pl", "rb"))
    y_test = pickle.load(open(path + "Toxic_y_test.pl", "rb"))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Toxic_X_train_sent = pickle.load(open(path + "Toxic_X_train_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test

def get_sof_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_sof.pl", "rb")).toarray()
    X_test = pickle.load(open(path + "X_test_sof.pl", "rb")).toarray()
    y_train = pickle.load(open(path + "y_train_sof.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_sof.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # X_train_sent = pickle.load(open(path + "X_train_sof_sent.pl", "rb"))

    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test    

def get_bert_imdb_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "X_train_bert_imdb.pl", "rb"))
    X_test = pickle.load(open(path + "X_test_bert_imdb.pl", "rb"))
    y_train = pickle.load(open(path + "y_train_bert_imdb.pl", "rb"))
    y_test = pickle.load(open(path + "y_test_bert_imdb.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test    

def get_squad_data():
    # path = "/content/drive/My Drive/Datasets/"
    X_train = pickle.load(open(path + "bert_squad_X_train.pl", "rb"))
    X_test = pickle.load(open(path + "bert_squad_X_test.pl", "rb"))
    y_train = pickle.load(open(path + "bert_squad_y_train.pl", "rb"))
    y_test = pickle.load(open(path + "bert_squad_y_test.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test 

def get_20_news_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ''
    X_train = pickle.load(open(path + "20_news_X_train.pl", "rb"))
    X_test = pickle.load(open(path + "20_news_x_val.pl", "rb"))
    y_train = pickle.load(open(path + "20_news_y_train.pl", "rb"))
    y_test = pickle.load(open(path + "20_news_y_val.pl", "rb"))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "20_news_X_train_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test

def get_ner_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ''
    X_train = pickle.load(open(path + "ner_X_train.pl", "rb"))
    X_test = pickle.load(open(path + "ner_X_test.pl", "rb"))
    y_train = pickle.load(open(path + "ner_y_train.pl", "rb"))
    y_test = pickle.load(open(path + "ner_y_test.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    
    # X_train_sent = pickle.load(open(path + "ner_X_train_sent.pl", "rb"))
    
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test    

def get_turkish_ner_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ''
    X_train = pickle.load(open(path + "turkish_ner_X_train.pl", "rb"))
    X_test = pickle.load(open(path + "turkish_ner_X_test.pl", "rb"))
    y_train = pickle.load(open(path + "turkish_ner_y_train.pl", "rb"))
    y_test = pickle.load(open(path + "turkish_ner_y_test.pl", "rb"))

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "turkish_ner_X_train_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test 



def get_ttc4900_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ''
    X_train = pickle.load(open(path + "ttc4900_X_train.pl", "rb"))
    X_test = pickle.load(open(path + "ttc4900_X_test.pl", "rb"))
    y_train = pickle.load(open(path + "ttc4900_y_train.pl", "rb"))
    y_test = pickle.load(open(path + "ttc4900_y_test.pl", "rb"))
    
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "ttc4900_X_train_sent.pl", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test     

def get_30columnists_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ''
    X_train = pickle.load(open(path + "30columnists_X_train.p", "rb"))
    X_test = pickle.load(open(path + "30columnists_X_test.p", "rb"))
    y_train = pickle.load(open(path + "30columnists_y_train.p", "rb"))
    y_test = pickle.load(open(path + "30columnists_y_test.p", "rb"))
    
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "30columnists_X_train_sent.p", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test 

def get_2500kose_data():
    # path = "/content/drive/My Drive/Datasets/"
    # path = ''
    X_train = pickle.load(open(path + "2500kose_X_train.p", "rb"))
    X_test = pickle.load(open(path + "2500kose_X_test.p", "rb"))
    y_train = pickle.load(open(path + "2500kose_y_train.p", "rb"))
    y_test = pickle.load(open(path + "2500kose_y_test.p", "rb"))
    
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    
    # X_train_sent = pickle.load(open(path + "2500kose_X_train_sent.p", "rb"))
    
    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test 


def get_ucl_data(name):
    X_train = pickle.load(open(name + "_X_train.p", "rb"))
    X_test = pickle.load(open(name + "_X_test.p", "rb"))
    y_train = pickle.load(open(name + "_y_train.p", "rb"))
    y_test = pickle.load(open(name + "_y_test.p", "rb"))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train.astype("float32"), y_train, X_test.astype("float32"), y_test

# def get_cross_val_data(name):
#     X_train = pickle.load(open(name + "_X.p", "rb"))
#     y_train = pickle.load(open(name + "_y.p", "rb"))
#     y_train = to_categorical(y_train)
#     return X_train.astype("float32"), y_train

from scipy.io import arff
from io import StringIO
def arff_to_numpy(path):
    with open(path) as f:
        content = f.read()
        data, meta = arff.loadarff(StringIO(content))

    dataset = np.array(data.tolist(), dtype=np.float)

    X = dataset[:, :-1]
    y = dataset[:, -1]
    y -= 1

    return X, y

def get_cross_val_data(name):
    # path = "/content/drive/My Drive/Datasets/"
    X, y = arff_to_numpy(path + name + ".arff")
    y = to_categorical(y)
    return X.astype("float32"), y

def read_labels(path_to_labels):
    with open(path_to_labels, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels - 1


def read_all_images(path_to_data):
    with open(path_to_data, "rb") as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images / 255.0
