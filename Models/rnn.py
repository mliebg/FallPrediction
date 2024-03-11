import numpy as np
import pickle
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import keras_tuner
from keras import layers
import tensorflow
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

# Load Data
print("load data")
ts = 200

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

##############################
# MODEL
##############################
def build_model():
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(13, input_shape=input_shape))
    #model.add(layers.Dropout(.4))
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
#print('searching best hypers')
#tuner.search(X_train_tensor, y_train, epochs=25, validation_data=(X_val_tensor, y_val))

print('fitting best model')
#best_hps = tuner.get_best_hyperparameters(1)
model = build_model()
model.Name = f'best LSTM {ts}'

# Defining Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'./Models/Evaluation/models/newbest_lstm{ts}.keras', save_best_only=True, monitor='val_loss'
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
#plt.close()
'''
# Top 10 best model with lowest FLOPs
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))

    concrete_func = concrete.get_concrete_function(

        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])

    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        return flops.total_float_ops


print('fitting best model with lowest FLOPs')
best_hps = tuner.get_best_hyperparameters(100)
best_hp = best_hps[0]
count = 0
for hp in best_hps:
    f_best = get_flops(build_model(best_hp))
    f_next = get_flops(build_model(hp))
    if f_best > f_next:
        best_hp = hp
        count += 1
    elif f_best == f_next:
        continue
    if count > 9:
        break

model = build_model(best_hp)
model.Name = f'best LSTM {ts} FLOPS'

# Defining Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'./Models/Evaluation/models/fbest_lstm{ts}.keras', save_best_only=True, monitor='val_loss'
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
    validation_data=(X_val_tensor, y_val),
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
# plt.close()



'''