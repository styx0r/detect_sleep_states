import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split

from data import import_data, reduce_dataset, extend_data_to_length

data_full, mappings = import_data()
data = reduce_dataset(data_full.copy())
# data = extend_data_to_length(data, mappings)

data.to_parquet("../data/data_preprocessed.parquet")
data = pd.read_parquet("../data/data_preprocessed.parquet")

# CNN with padding
nr_series_ids = data["series_id"].unique().size

data["step"] = data.groupby("series_id").cumcount()
data["step"] = np.uint32(data["step"])

STEPS = 1440
SAMPLE_SIZE_PER_SERIES_ID = 2048
indices_per_series_id = {
    series_id: list(
        set(
            [
                random.randint(0, max_steps - STEPS)
                for i in range(SAMPLE_SIZE_PER_SERIES_ID)
            ]
        )
    )
    for series_id, max_steps in data.groupby("series_id")["step"].max().items()
    if max_steps >= STEPS
}

X = (
    data[data["series_id"].isin(indices_per_series_id.keys())]
    .groupby("series_id")
    .apply(
        lambda d: pd.concat(
            [
                d[step : (step + STEPS)].assign(step_group=step)
                for step in indices_per_series_id[d["series_id"].iloc[0]]
            ]
        )
    )
    .reset_index(drop=True)
)

Y = np.array(X.groupby(["series_id", "step_group"])["event"].apply(list).tolist())

features = ["enmo_mean"]  # , "anglez_mean"]  # Add all your feature names here
X = np.stack(
    [
        np.array(X.groupby(["series_id", "step_group"])[feature].apply(list).tolist())
        for feature in features
    ],
    axis=-1,
)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Conv1D(
        256, 360, activation="relu", input_shape=(x_train.shape[1], x_train.shape[2])
    )
)
model.add(tf.keras.layers.MaxPooling1D(30))
model.add(tf.keras.layers.Conv1D(512, 30, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000, activation="relu"))
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

BATCH_SIZE = 128

with tf.device("/gpu:0"):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=BATCH_SIZE,
        callbacks=tf.keras.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True
        ),
    )

for k, v in history.history.items():
    _ = plt.plot(v, label=k)
_ = plt.legend()

index = 10
x_pred = x_train[[index]]
prediction = model.predict(x_pred, BATCH_SIZE)
prediction = pd.DataFrame(
    prediction.reshape((prediction.shape[0] * prediction.shape[1]))
)
prediction = prediction.rename(columns={0: "prediction"})

# prediction["enmo_mean"] = x_train[[index]].squeeze()
prediction["event"] = y_train[[index]].squeeze()
prediction["enmo_mean"] = x_train[[index]][:, :, 0].squeeze()
# prediction["enmo_var"] = x_train[[index]][:,:,1].squeeze()

# prediction[["event", "enmo_mean"]][:10000].plot()
prediction.plot()
((prediction["prediction"] - prediction["event"]) ** 2).mean()

# mse = []
# for index in range(10000):
#     x_pred = x_train[[index]]
#     prediction = model.predict(x_pred, BATCH_SIZE)
#     prediction = pd.DataFrame(
#         prediction.reshape((prediction.shape[0] * prediction.shape[1]))
#     )
#     prediction = prediction.rename(columns={0: "prediction"})

#     # prediction["enmo_mean"] = x_train[[index]].squeeze()
#     prediction["event"] = y_train[[index]].squeeze()
#     prediction["enmo_mean"] = x_train[[index]][:, :, 0].squeeze()
#     # prediction["enmo_var"] = x_train[[index]][:,:,1].squeeze()

#     # prediction[["event", "enmo_mean"]][:10000].plot()
#     # prediction.plot()
#     mse.append(((prediction["prediction"] - prediction["event"]) ** 2).mean())


# take a complete sample and
# - create step_size batches
# - make predictions
# - create full sample with prediction again
BATCH_RESOLUTION = 144
patient = data[data.series_id == 2].reset_index(drop=True)
patient_batches = []
r = 0
while r <= patient.shape[0] - STEPS:
    patient_batches.append(
        patient[r : r + STEPS].assign(batch=int(r / BATCH_RESOLUTION))
    )
    r += BATCH_RESOLUTION

patient_batches = pd.concat(patient_batches)

X_pred = np.stack(
    [
        np.array(
            patient_batches.groupby(["series_id", "batch"])[feature]
            .apply(list)
            .tolist()
        )
        for feature in features
    ],
    axis=-1,
)


predictions = np.zeros((patient.shape[0], patient_batches.batch.max() + 1)) - 1
for batch_index in range(0, X_pred.shape[0]):
    prediction = model.predict(X_pred[[batch_index]], BATCH_SIZE)
    predictions[
        (batch_index * BATCH_RESOLUTION) : ((batch_index * BATCH_RESOLUTION) + STEPS),
        batch_index,
    ] = prediction.squeeze()

masked_predictions = np.ma.masked_equal(predictions, -1)
patient["prediction"] = np.ma.mean(masked_predictions, axis=1).data
patient["prediction_std"] = np.ma.std(masked_predictions, axis=1).data

patient[features + ["event", "prediction"]][0:2000].plot()
