import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
import keras_tuner

##############################
# MultiLayer Perceptron
##############################

ts = 100  # or 200
if ts == 200:
    max_units = 160
elif ts == 100:
    max_units = 256

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
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f'units_{i}', min_value=16, max_value=max_units, step=16),
                activation=hp.Choice('activation', ['sigmoid', 'tanh', 'relu'])
            )
        )
        model.add(layers.Dropout(0.3))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


# Tune Hyperparameters
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=100,
    executions_per_trial=1,
    directory='./Models',
    project_name=f'mlp{ts}_tuning'
)
print('searching best hypers')
tuner.search(X_train_tensor, y_train, epochs=50, validation_data=(X_val_tensor, y_val))
best_hps = tuner.get_best_hyperparameters(1)
model = build_model(best_hps[0])
model.Name = f'best MLP {ts}'

# Defining Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'./Models/Evaluation/keras-models/best_mlp{ts}.keras', save_best_only=True, monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, verbose=1
    )
]

# Fitting the model
print('fitting best model')
history = model.fit(
    X_train_tensor,
    y_train,
    epochs=1000,
    batch_size=16,
    callbacks=callbacks,
    validation_data=(X_val_tensor, y_val),
    verbose=1)

# Plot training model loss
metric = 'sparse_categorical_accuracy'
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history['val_' + metric])
plt.title(f'MLP{ts} Training')
plt.ylabel('SCA', fontsize='large')
plt.xlabel("Epoche", fontsize='large')
plt.legend(['Trainingsset SCA', 'Validierungsset SCA'], loc='best')
plt.xlim(0, 1000)
plt.ylim(0, 1)
plt.show()
