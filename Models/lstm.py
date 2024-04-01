import numpy as np
import pickle
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import keras_tuner
from keras import layers

##############################
# Long Short-Term Memory
##############################

ts = 200
if ts == 200:
    max_units = 13
elif ts == 100:
    max_units = 20

# Load Data
print('load data')
# Constructing the input
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_xtrain.pkl', 'rb') as f:
    X_train = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_xval.pkl', 'rb') as f:
    X_val = pickle.load(f)
f.close()

# Constructing the output
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_ytrain.pkl', 'rb') as f:
    y_train = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_yval.pkl', 'rb') as f:
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
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(units=hp.Int('units', min_value=5, max_value=max_units, step=1), input_shape=input_shape))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    return model


# Tune Hyperparameters
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=10,  # LSTM doesn't need 100 Trials bc there is only one parameter to tune
    executions_per_trial=1,
    directory='./Models',
    project_name=f'lstm{ts}_tuning'
)
print('searching best hypers')
tuner.search(X_train_tensor, y_train, epochs=25, validation_data=(X_val_tensor, y_val))

print('fitting best model')
best_hps = tuner.get_best_hyperparameters(1)
model = build_model(best_hps[0])
model.Name = f'best LSTM {ts}'

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
    batch_size=16,
    callbacks=callbacks,
    validation_data=(X_val_tensor, y_val),
    verbose=1)

# Plot model loss
metric = 'sparse_categorical_accuracy'
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history['val_' + metric])
plt.title('LSTM Training' + metric)
plt.ylabel('SCA', fontsize='large')
plt.xlabel('Epoche', fontsize='large')
plt.legend(['Trainingsset SCA', 'Validierungsset SCA'], loc='best')
plt.show()
