import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
# from transformers import TFBertModel

def get_model(name, class_num = 2):
    if name == "cifar10":
        return get_cifar10_model()
    elif name == "cifar100":
        return get_cifar100_model()
    elif name == "fashion":
        return get_fashion_model()
    elif name == "mnist":
        return get_mnist_model()
    elif name == "kmnist":
        return get_mnist_model()
    elif name == "big_cifar10":
        return get_cifar10_big_model()
    elif name == "big_cifar100":
        return get_cifar100_big_model()
    elif name == "big_fashion":
        return get_fashion_big_model()
    elif name == "big_mnist":
        return get_mnist_big_model()
    elif name == "big_kmnist":
        return get_mnist_big_model()
    elif name == "beans":
        return get_beans_model()
    elif name == "stl_10":
        return get_stl_10_model()
    elif name == "svhn":
        return get_cifar10_model()
    elif name == "oxford_flowers":
        return get_oxford_flowers_model()
    elif name == "tf_flowers":
        return get_tf_flowers_model()
    elif name == "imdb":
        return get_imdb_model()
    elif name == "imdb1":
        return get_imdb1_model()
    elif name == "sarcasm":
        return get_sarcasm_model()
    elif name == "hotel":
        return get_hotel_model()
    elif name == 'stweet':
        return get_stweet_model()
    elif name == 'qpair':
        return get_qpair_model()
    elif name == 'ctweet':
        return get_ctweet_model()
    elif name == 'food':
        return get_food_model()
    elif name == 'sof':
        return get_sof_model()
    elif name == 'reddit':
        return get_reddit_model()
    elif name == "toxic":
        return get_toxic_model()
    elif name == 'reuters':
        return get_reuters_model()
    elif name == "20_news":
        return get_20_news_model()
    elif name == 'bert_imdb':
        return get_bert_imdb_model()
    elif name == 'squad':
        return get_squad_model()
    elif name == 'ner':
        return get_ner_model()
    elif name == 'turkish_ner':
        return get_turkish_ner_model()
    elif name == 'ttc4900':
        return get_ttc4900_model()
    elif name == '30columnists':
        return get_30columnists_model()
    elif name == '2500kose':
        return get_2500kose_model()
    else:
        return get_ucl_model(name, class_num)


# def get_cifar10_model():
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(10, activation="softmax"))
#     opt = 'adam'
#     model.compile(
#         optimizer=opt,
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     # model.summary()

#     return model

# def get_cifar10_model():
#     # model = tf.keras.applications.ResNet50(
#     # weights=None,
#     # input_shape=(32, 32, 3),
#     # include_top=True,
#     # classes=10)

#     # opt = "adam"
#     # model.compile(
#     #     optimizer=opt,
#     #     loss="categorical_crossentropy",
#     #     metrics=["accuracy"],
#     # )

#     # # model.summary()
    
#     model = tf.keras.applications.MobileNetV2(
#         input_shape=(32, 32, 3),
#         alpha=1.0,
#         include_top=True,
#         weights=None,
#         input_tensor=None,
#         pooling=None,
#         classes=10,
#         classifier_activation="softmax",
#     )

#     opt = "adam"
#     model.compile(
#         optimizer=opt,
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     # model.summary()


#     return model

def get_cifar10_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Flatten())
    # model.add(layers.Dense(64))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation('relu'))
    model.add(layers.Dense(10, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    return model

def get_cifar10_big_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10

    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    model.summary()
    
    return model

def get_oxford_flowers_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(102, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model

# def get_cifar100_model():
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(100, activation="softmax"))
#     # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
#     opt = "adam"
#     model.compile(
#         optimizer=opt,
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model


def get_cifar100_big_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation='softmax'))    # num_classes = 10

    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model

def get_cifar100_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
              
    model.add(layers.Dense(100, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = 'adam'
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model


# def get_fashion_model():
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(10, activation="softmax"))
#     # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
#     opt = "adam"
#     model.compile(
#         optimizer=opt,
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model


def get_fashion_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(10, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = 'adam'
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model

def get_fashion_big_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10

    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model
    
def get_mnist_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(10, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = 'adam'
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model

def get_mnist_big_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28 ,28 ,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10

    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model
    
def get_stl_10_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(96, 96, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    # model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = 'adam'
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model

def get_beans_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(500, 500, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = 'adam'
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model

def get_tf_flowers_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(442, 1024, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model

def get_aircraft_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(73, 141, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="softmax"))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = "adam"
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    return model


def get_imdb_model():

    # Embedding
    embedding_size = 50
    max_features = 5000
    maxlen = 400

    # Convolution
    kernel_size = 5
    pool_size = 4
    filters = 64

    # LSTM
    lstm_output_size = 70

    model = models.Sequential()

    # Embedding layer
    model.add(layers.Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(layers.Dropout(0.25))

    # Convolutional layer
    model.add(
        layers.Conv1D(
            filters, kernel_size, padding="valid", activation="relu", strides=1
        )
    )
    model.add(layers.MaxPooling1D(pool_size=pool_size))

    # LSTM layer
    model.add(layers.LSTM(lstm_output_size))

    # Squash
    model.add(layers.Dense(2))
    model.add(layers.Activation("sigmoid"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # model.summary()

    return model

def get_imdb1_model():
    max_features = 6000
    maxlen = 130
    embed_size = 128
    model = models.Sequential()
    model.add(layers.Embedding(max_features, embed_size))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences = True)))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(20, activation="relu"))
    model.add(layers.Dropout(0.05))
    # model.add(Dense(1, activation="sigmoid"))
    model.add(layers.Dense(2, activation="softmax"))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # batch_size = 100
    # epochs = 6
    
    return model    


# def get_stl_10_model():
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(96, 96, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation="relu"))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(10, activation="softmax"))
#     # opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
#     opt = "adam"
#     model.compile(
#         optimizer=opt,
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model


def get_sarcasm_model():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 10, input_length=100))
    model.add(layers.Bidirectional(layers.GRU(32, return_sequences=True)))
    model.add(layers.GlobalAvgPool1D())
    model.add(layers.Dense(500, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation="softmax"))

    # %%
    # model.summary()

    # %%
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def get_ucl_model(name, class_num):
    model = models.Sequential()

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    # model.add(layers.Dense(8, activation="relu"))

    model.add(layers.Dense(class_num, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def get_hotel_model():
    embedding_dim = 16
    units = 76
    vocab_size = 49536
    input_length = 350

    model = tf.keras.Sequential(
        [
            layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
            layers.Bidirectional(layers.LSTM(units, return_sequences=True)),
            # L.LSTM(units,return_sequences=True),
            layers.Conv1D(64, 3),
            layers.MaxPool1D(),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # model.summary()

    return model


def get_toxic_model():

    path = "/content/drive/My Drive/Datasets/"

    embedding_matrix = pickle.load(open(path + "toxic_embedding_matrix.pl", "rb"))

    max_features = 20000
    max_text_length = 400
    embedding_dim = 100

    model = models.Sequential()
    model.add(
        layers.Embedding(
            max_features,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )
    )
    model.add(layers.Dropout(0.2))

    # %%
    """
    ### Build the Model
    """

    # %%
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    # %%
    model.add(layers.Conv1D(filters, kernel_size, padding="valid"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters, 5, padding="valid", activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(hidden_dims, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation="sigmoid"))
    # model.summary()

    # %%
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model

def get_reuters_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    
    model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_reddit_model():
    embedding_dim = 16
    max_length = 100
    vocab_size = 8000
    model = models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    # num_epochs = 2
    # batch_size = 2048
    
    return model    

def get_stweet_model():
    
    path = "/content/drive/My Drive/Datasets/"
    
    embedding_matrix = pickle.load(open(path + 'embedding_matrix_tweet.pl','rb'))
    
    embedding_layer = tf.keras.layers.Embedding(290575,
                                              300,
                                              weights=[embedding_matrix],
                                              input_length=30,
                                              trainable=False)
    
    sequence_input = layers.Input(shape=(30,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)
    x = layers.SpatialDropout1D(0.2)(embedding_sequences)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(sequence_input, outputs)
    model.compile(optimizer='adam', loss='CategoricalCrossentropy',
                  metrics=['accuracy'])
    return model    

def get_qpair_model():
    embedding_size = 128

    inp1 = layers.Input(shape=(100,))
    inp2 = layers.Input(shape=(100,))
    
    x1 = layers.Embedding(6000, embedding_size)(inp1)
    x2 = layers.Embedding(6000, embedding_size)(inp2)
    
    x3 = layers.Bidirectional(layers.LSTM(32, return_sequences = True))(x1)
    x4 = layers.Bidirectional(layers.LSTM(32, return_sequences = True))(x2)
    
    x5 = layers.GlobalMaxPool1D()(x3)
    x6 = layers.GlobalMaxPool1D()(x4)
    
    x7 =  layers.dot([x5, x6], axes=1)
    
    x8 = layers.Dense(40, activation='relu')(x7)
    x9 = layers.Dropout(0.05)(x8)
    x10 = layers.Dense(10, activation='relu')(x9)
    output = layers.Dense(2, activation="softmax")(x10)
    
    model = models.Model(inputs=[inp1, inp2], outputs=output)
    model.compile(loss='CategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])
    # batch_size = 100
    # epochs = 3
    return model    

def get_ctweet_model():
    # hyper parameters
    # EPOCHS = 5
    # BATCH_SIZE = 32
    embedding_dim = 16
    units = 256
    
    model = tf.keras.Sequential([
        layers.Embedding(36117, embedding_dim, input_length=54),
        layers.Bidirectional(layers.LSTM(units,return_sequences=True)),
        layers.GlobalMaxPool1D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(3, activation="softmax")
    ])
    
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',metrics=['accuracy']
                 )
    
    # model.summary()

    return model

def get_sof_model():
    model = models.Sequential()
    
    model.add(layers.Dense(128, activation='tanh'))
    model.add(layers.Dense(128, activation='tanh'))
    model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model    

def get_food_model():
    top_words = 6000
    # tokenizer = Tokenizer(num_words=top_words)
    # tokenizer.fit_on_texts(train_df['Text'])
    # list_tokenized_train = tokenizer.texts_to_sequences(train_df['Text'])
    
    max_review_length = 130
    # X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)
    # y_train = train_df['Score']
    embedding_vecor_length = 32
    model = models.Sequential()
    model.add(layers.Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
    model.add(layers.LSTM(100))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    
    # batch_size = 64
    # epochs = 5
    
    return model

def get_20_news_model():

    path = "/content/drive/My Drive/Datasets/"
    # path = ''
    embedding_matrix = pickle.load(open(path + "20_news_embedding_matrix.pl", "rb"))

    model = models.Sequential()
    model.add(
        layers.Embedding(
            20002,
            100,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )
    )

    model.add(layers.Conv1D(128, 5, activation="relu"))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(128, 5, activation="relu"))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(128, 5, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(20, activation="softmax"))

    # model.summary()

    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    return model

def get_bert_imdb_model():
    from transformers import TFBertModel
    max_len = 512

    ## BERT encoder
    # encoder = TFBertModel.from_pretrained("bert-base-uncased")

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    
    

    # inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
    
    inputs = [input_ids, token_type_ids, attention_mask]
    
    # name = "distilbert-base-uncased"
    name = "bert-base-cased"

    # m = TFAutoModelForSequenceClassification.from_pretrained(name)

    # bert = m.layers[0]

    bert = TFBertModel.from_pretrained(name)
    
    bert.trainable = True
    
    bert_outputs  = bert(
        input_ids = input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )

    last_hidden_states = bert_outputs.last_hidden_state
    avg = layers.GlobalAveragePooling1D()(last_hidden_states)
    avg = tf.keras.layers.Dense(128, activation='relu')(avg)
    output = layers.Dense(2, activation="softmax")(avg)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(lr=5e-5)
    # optimizer = 'adam'
    model.compile(optimizer=optimizer, loss=[loss], metrics = ['accuracy'])
    return model

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_ner_model():
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    
    n_token = 35179
    n_tag = 17
    vocab_size = 35179
    
    input_dim = 104
    
    inputs = layers.Input(shape=(input_dim,))
    
    embedding_layer = TokenAndPositionEmbedding(input_dim, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    outputs = layers.TimeDistributed(layers.Dense(n_tag, activation="softmax"))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # model.summary()

    return model    

def get_turkish_ner_model():
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    
    n_token = 123086
    n_tag = 9
    vocab_size = 123086
    
    input_dim = 50
    
    inputs = layers.Input(shape=(input_dim,))
    
    embedding_layer = TokenAndPositionEmbedding(input_dim, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    outputs = layers.TimeDistributed(layers.Dense(n_tag, activation="softmax"))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # model.summary()

    return model

def get_ttc4900_model():
    vocab_size = 94644
    embedding_dim=16

    model = tf.keras.Sequential([
        
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=100),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    adam= tf.keras.optimizers.Adam(lr=0.01)
    
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    # model.summary()
    
    return model

def get_30columnists_model():
    embedding_dim = 150
    
    vocab_size = 28638
    
    max_len = 500
    
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                               output_dim=embedding_dim, 
                               input_length=max_len))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(layers.Dense(30, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    
    return model

def get_2500kose_model():
    embedding_dim = 150
    vocab_size = 124255
    max_len = 1000

    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                               output_dim=embedding_dim, 
                               input_length=max_len))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    
    return model

def get_squad_model():
    from transformers import TFBertModel
    max_len = 384
    ## BERT encoder
    encoder = TFBertModel.from_pretrained("bert-base-uncased")

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(tf.keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(tf.keras.activations.softmax)(end_logits)

    model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model