import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Importing the data
df = pd.read_csv('./Models/basicAnnTest/random_data.csv')

# split the data into train and test set
train, test = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True)

# Constructing the input
x = np.column_stack((train.X.values, train.Y.values))
y = train.Color.values

# Defining the model
model = keras.Sequential([
    keras.layers.Dense(4, input_shape=(2,), activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# fitting the model
model.fit(x, y, epochs=10, batch_size=8)

# Evaluating the model
x = np.column_stack((test.X.values, test.X.values))
y = test.Color.values
model.evaluate(x, y, batch_size=8)

