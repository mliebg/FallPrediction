import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

##############################
# MultiLayer Perceptron
##############################

# Load Data
print("load data")
ts = 100
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

#val_ds = tf.data.Dataset.from_tensor_slices(X_val_tensor, y_val)
# Defining the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dropout(.1),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Defining Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'./Models/Evaluation/models/best_mlp{ts}.keras', save_best_only=True, monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, verbose=1
    )
]

# Fitting the model
history = model.fit(
    X_train_tensor,
    y_train,
    epochs=1000,
    batch_size=16,
    callbacks=callbacks,
    #validation_data=val_ds,
    validation_split=0.2,
    verbose=1)

# Plot model loss
metric = 'sparse_categorical_accuracy'
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history['val_' + metric])
plt.title('model ' + metric)
plt.ylabel(metric, fontsize='large')
plt.xlabel("epoch", fontsize='large')
plt.legend(['train', 'val'], loc='best')
plt.show()
#plt.close()
