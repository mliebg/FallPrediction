import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os

# Importing the data

dir_path = './DataAcquisition/DataLabeling/simpleLabel-isFallen/data'
csv_list = []

for csv in os.listdir(dir_path):
    if csv.endswith(".csv"):
        csv_path = os.path.join(dir_path, csv)
        dataframe = pd.read_csv(csv_path)
        csv_list.append(dataframe)

df = pd.concat(csv_list, axis=0, ignore_index=True)

# split the data into train and test set
train, test = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True)

# Constructing the input
x = np.column_stack((train.gyroYaw.values,
                     train.gyroPitch.values,
                     train.gyroRoll.values,
                     train.accelX.values,
                     train.accelY.values,
                     train.accelZ.values,
                     ))
y = train.isFallen.values

# Defining the model
model = keras.Sequential([
    keras.layers.Dense(18, input_shape=(6,), activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# fitting the model
model.fit(x, y, epochs=10, batch_size=8)

# Evaluating the model
x = np.column_stack((test.gyroYaw.values,
                     test.gyroPitch.values,
                     test.gyroRoll.values,
                     test.accelX.values,
                     test.accelY.values,
                     test.accelZ.values,
                     ))
y = test.isFallen.values
model.evaluate(x, y, batch_size=8)

model.save('./Models/simpleANN/simpleAnn.keras')

