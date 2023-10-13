import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split

from data import import_data, reduce_dataset, extend_data_to_length

data_full, mappings = import_data()
data = reduce_dataset(data_full.copy())
data = extend_data_to_length(data, mappings)

data.to_parquet("../data/data_preprocessed.parquet")
data = pd.read_parquet("../data/data_preprocessed.parquet")

# CNN with padding
nr_series_ids = data["series_id"].unique().size
values_per_series_id = data.groupby("series_id").apply(lambda s: s.shape[0]).unique()[0]

data["step"] = data.groupby("series_id").cumcount()

features = ["enmo_mean"]  # , "anglez_mean"]  # Add all your feature names here
X = np.stack(
    [
        np.array(data.groupby("series_id")[feature].apply(list).tolist())
        for feature in features
    ],
    axis=-1,
)

Y = np.array(data.groupby("series_id")["event"].apply(list).tolist())

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Conv1D(
        32, 30, activation="relu", input_shape=(x_train.shape[1], x_train.shape[2])
    )
)
# model.add(tf.keras.layers.MaxPooling1D(30))
model.add(tf.keras.layers.Conv1D(64, 30, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(
    tf.keras.layers.Dense(x_train.shape[1], activation="sigmoid")
)  # Adjust based on the problem
# Compile the model
model.compile(
    # optimizer=optimizers.Adam(0.1, weight_decay=0.001),
    optimizer=tf.keras.optimizers.legacy.Adam(decay=0.005),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    # metrics=["MAE"],
    metrics=["MSE", "MAE"],
)
model.summary()

with tf.device("/gpu:0"):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=300,
        batch_size=64,
        callbacks=tf.keras.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True
        ),
    )

for k, v in history.history.items():
    _ = plt.plot(v, label=k)
_ = plt.legend()

index = 100
x_pred = x_train[[index]]
prediction = model.predict(x_pred, 64)
prediction = pd.DataFrame(
    prediction.reshape((prediction.shape[0] * prediction.shape[1]))
)
prediction = prediction.rename(columns={0: "prediction"})

# prediction["enmo_mean"] = x_train[[index]].squeeze()
prediction["event"] = y_train[[index]].squeeze()
prediction["enmo_mean"] = x_train[[index]][:, :, 0].squeeze()
# prediction["anglez_mean"] = x_test[[index]][:,:,1].squeeze()

prediction[["event", "enmo_mean"]][:10000].plot()
prediction[:10000].plot()
