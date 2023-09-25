import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from data import import_data, reduce_dataset, extend_data_to_length

import tensorflow as tf

from sklearn.model_selection import train_test_split

# Set the number of CPU threads
# num_threads = str(multiprocessing.cpu_count())
# os.environ["OMP_NUM_THREADS"] = num_threads
# os.environ["TF_NUM_INTRAOP_THREADS"] = num_threads
# os.environ["TF_NUM_INTEROP_THREADS"] = num_threads

# tf.config.threading.set_inter_op_parallelism_threads(int(num_threads))
# tf.config.threading.set_intra_op_parallelism_threads(int(num_threads))
# tf.config.set_soft_device_placement(True)

# TODO: think about normalization, does not work probably using the mean or the median
# maybe something with a quantile, but what value?
# idx = 4
# (data_full[data_full.series_id == idx]['enmo'] / data_full[data_full.series_id == idx]['enmo'].quantile(0.99)).plot()
# idx = 4
# (data_full[data_full.series_id == idx]['enmo'] / data_full[data_full.series_id == idx]['enmo'].median()).plot()
# (data_full[data_full.series_id == 0]['enmo'] / data_full[data_full.series_id == 0]['enmo'].median()).plot()
# tmp = data_full.groupby("series_id")["enmo"].apply(lambda es: es / es.median())
# data_full["enmo"] = tmp.to_numpy()
# here we should normalize each measure to the mean to make different subjects more comparable

data_full, mappings = import_data()
data = reduce_dataset(data_full.copy())
data = extend_data_to_length(data, mappings)

data.to_parquet("../data/data_preprocessed.parquet")
data = pd.read_parquet("../data/data_preprocessed.parquet")

# Todo
# try normalize anglez against mean
# and enmo against mean for all values > 0

# CNN with padding
nr_series_ids = data["series_id"].unique().size
values_per_series_id = data.groupby("series_id").apply(lambda s: s.shape[0]).unique()[0]

data["step"] = data.groupby("series_id").cumcount()

features = [
    "enmo_mean",
    # "enmo_var",
    # "anglez_mean",
    # "anglez_var",
]  # Add all your feature names here
X = np.stack(
    [
        np.array(data.groupby("series_id")[feature].apply(list).tolist())
        for feature in features
    ],
    axis=-1,
)

Y = np.array(data.groupby("series_id")["event"].apply(list).tolist())

# train = data["enmo_mean"].to_numpy().reshape((values_per_series_id, nr_series_ids)).T
# train = data["event"].to_numpy().reshape((values_per_series_id, nr_series_ids)).T

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

x_train = x_train.squeeze()

with tf.device("/gpu:0"):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv1D(
            30,
            # (720, x_train.shape[2]),
            120,
            padding="same",
            activation="relu",
            # input_shape=(x_train.shape[1], x_train.shape[2], 1),
            input_shape=(x_train.shape[1], 1),
        )
    )
    # model.add(
    #     tf.keras.layers.Conv1D(
    #         1,
    #         # (720, x_train.shape[2]),
    #         720,
    #         padding="same",
    #         activation="relu",
    #     )
    # )
    # model.add(
    #     tf.keras.layers.Conv1D(
    #         # 1, (120, x_train.shape[2]),
    #         1,
    #         120,
    #         padding="same",
    #         activation="relu",
    #     )
    # )
    # model.add(
    #     tf.keras.layers.Conv1D(
    #         # 1, (30, x_train.shape[2]),
    #         1,
    #         30,
    #         padding="same",
    #         activation="sigmoid",
    #     )
    # )

    # Global Average Pooling
    # model.add(tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first"))
    # model.add(tf.keras.layers.Flatten())

    # model.add(tf.keras.layers.Conv2D(1, (x_train.shape[1], 1), padding="valid", activation="sigmoid"))

    # Dense layer to match the target shape
    # model.add(tf.keras.layers.Dense(units=125280, activation="sigmoid"))
    # Add dense layers on top
    # model.add(tf.keras.layers.Flatten())
    # Add a fully connected (dense) layer
    # model.add(tf.keras.layers.Dense(units=720, activation="relu"))

    # Final dense layer to match the target shape
    model.add(
        tf.keras.layers.Dense(units=720, activation="relu")
    )  # Adjust activation if needed

    model.add(
        tf.keras.layers.Dense(units=1, activation="sigmoid")
    )  # Adjust activation if needed

    # Compile the model
    model.compile(
        # optimizer=optimizers.Adam(0.1, weight_decay=0.001),
        optimizer=tf.keras.optimizers.legacy.Adam(decay=0.005),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["MSE", "MAE"],
    )
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=tf.keras.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True
        ),
    )

for k, v in history.history.items():
    _ = plt.plot(v, label=k)
_ = plt.legend()

index = 0
x_pred = x_train[[index]]
prediction = model.predict(x_pred, 16)
prediction = pd.DataFrame(
    prediction.reshape((prediction.shape[0] * prediction.shape[1]))
)
prediction = prediction.rename(columns={0: "prediction"})

prediction["enmo_mean"] = x_train[[index]].squeeze()
prediction["event"] = y_train[[index]].squeeze()

prediction[:10000].plot()

data_pred = data[data.series_id == 0].copy()
prediction = pd.DataFrame(
    prediction.reshape((prediction.shape[0] * prediction.shape[1]))
)
prediction = prediction.rename(columns={0: "prediction"})

data_pred["prediction"] = prediction

data_pred.iloc[2000:12000][["enmo_var", "event", "prediction"]].plot()

# Some plotting

# mean enmo patient vise
data[data["event"] == 0].groupby(["series_id"])["enmo"].mean().hist(bins=50)
data[data["event"] == 0].groupby(["series_id"])["enmo"].max().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["enmo"].mean().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["enmo"].max().hist(bins=50)

data[data["event"] == 0].groupby(["series_id"])["anglez"].mean().hist(bins=50)
data[data["event"] == 0].groupby(["series_id"])["anglez"].max().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["anglez"].mean().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["anglez"].max().hist(bins=50)

# ToDo: is max and avg correlated? Does it make sense to normalize?
# -> we could also add as feature and calculate feature importance

sid = data[data.series_id == 1].copy().reset_index(drop=True)
sid = sid[sid.event != -1]
sid["enmo_norm"] = sid["enmo"] / sid["enmo"].max()
sid["anglez_norm"] = (sid["anglez"] - sid["anglez"].min()) / (
    sid["anglez"].max() - sid["anglez"].min()
)
sid.loc[:40000][["event", "enmo_norm", "anglez_norm"]].plot()


# Feature engineering
# Normalize values grouped by series_id
# take differences of two consecutive timepoints


# To try:
# https://github.com/angus924/rocket

# https://pyts.readthedocs.io/en/stable/generated/pyts.classification.TimeSeriesForest.html#pyts.classification.TimeSeriesForest.fit

# xgboost and efficient way to extract data from dataset


##################### general CNN ##################
# RUN CPU only
import tensorflow as tf

tf.config.list_physical_devices()
tf.config.list_logical_devices()

# with tf.device("/cpu:0"):
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (
    test_images,
    test_labels,
) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

model.summary()

model.compile(
    optimizer=optimizers.legacy.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_images,
    train_labels,
    epochs=20,
    batch_size=128,
    validation_data=(test_images, test_labels),
)

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.legend(loc="lower right")

test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=64, verbose=2)


import tensorflow as tf

with tf.device("/cpu:0"):
    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)


# FIX ISSUES
pd.Series(
    [
        sum([abs(Y[j][i] - Y[j][i - 1]) for i in range(1, Y.shape[1])])
        for j in range(Y.shape[0])
    ]
).hist(bins=100)
data_full.groupby("series_id")["event"].apply(lambda x: (x.diff() != 0).sum()).hist(
    bins=100
)
(
    train_series.groupby("series_id")["step"].apply(lambda x: x.max() - x.min())
    * 5
    / 60
    / 60
    / 24
).hist(bins=100)
