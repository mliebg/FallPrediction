import numpy as np
import pickle
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import tensorflow

#######################
# Load Data
#######################
print("load data")
ts = 100
# Constructing the input
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_xtrain.pkl', 'rb') as f:
    X_train = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_xval.pkl', 'rb') as f:
    X_val = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_xtest.pkl', 'rb') as f:
    X_test = pickle.load(f)
f.close()

# Constructing the output
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_ytrain.pkl', 'rb') as f:
    y_train = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_yval.pkl', 'rb') as f:
    y_val = pickle.load(f)
f.close()
with open(f'./DataPreprocessing/ts{ts}/ts{ts}_ytest.pkl', 'rb') as f:
    y_test = pickle.load(f)
f.close()

num_classes = len(np.unique(y_train))

# Convert X (Dataframes) into tensor
arrays = [df.values for df in X_train]
stacked = tf.stack(arrays)

X_train_tensor = tf.convert_to_tensor(stacked)
input_shape = X_train_tensor.shape[1:]

arrays = [df.values for df in X_test]
stacked = tf.stack(arrays)

X_test_tensor = tf.convert_to_tensor(stacked)

##############################
# MODEL
##############################
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(200),
    #tf.keras.layers.SimpleRNN(200),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Defining Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_lstm.keras', save_best_only=True, monitor='val_loss'
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
    validation_split=0.2,
    verbose=1)

# Evaluating the model
print('\nMODEL EVALUATION')
model = keras.models.load_model('best_lstm.keras')
test_loss, test_acc = model.evaluate(X_test_tensor, y_test)#batchsize = 8

print('Test accuracy', test_acc)
print('Test loss', test_loss)

# Model summary
print('\nMODEL SUMMARY')
model.summary()

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


