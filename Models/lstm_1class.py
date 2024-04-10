import numpy as np
import pickle
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import layers

##############################
# Long Short-Term Memory
##############################

ts = 50

# Load Data
print('load data')
# Constructing the input
with open(f'./DataPreprocessing/ts{ts}_1class/ts{ts}_xtrain.pkl', 'rb') as f:
    X_train = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}_1class/ts{ts}_xval.pkl', 'rb') as f:
    X_val = pickle.load(f)
f.close()
# Constructing the output
with open(f'./DataPreprocessing/ts{ts}_1class/ts{ts}_ytrain.pkl', 'rb') as f:
    y_train = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}_1class/ts{ts}_yval.pkl', 'rb') as f:
    y_val = pickle.load(f)
f.close()

num_classes = len(np.unique(y_train))

# Convert X (Dataframes) into tensor
arrays = [df.values for df in X_train]
stacked = tf.stack(arrays)

X_train_tensor = tf.convert_to_tensor(stacked)
input_shape = X_train_tensor.shape[1:]

arrays = [df.values for df in X_val]
stacked = tf.stack(arrays)

X_val_tensor = tf.convert_to_tensor(stacked)


# Defining the model
def build_model():
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(32, input_shape=input_shape))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    # Compile
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


model = build_model

# Defining Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'./Models/Evaluation/keras-models/best_lstm{ts}.keras', save_best_only=True, monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, verbose=1
    )
]

history = model.fit(
    X_train_tensor,
    y_train,
    epochs=1000,
    batch_size=32,
    callbacks=callbacks,
    validation_data=(X_val_tensor, y_val),
    verbose=1)

# Plot model loss
metric = 'categorical_accuracy'
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history['val_' + metric])
plt.title('LSTM Training' + metric)
plt.ylabel('SCA', fontsize='large')
plt.xlabel('Epoche', fontsize='large')
plt.legend(['Trainingsset SCA', 'Validierungsset SCA'], loc='best')
plt.show()
