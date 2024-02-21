# Multilayer Perceptron

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

##############################
# PREPROCESSING
##############################
# Import Data
src_dir_path = './DataAcquisition/DataLabeling/FullyLabeled/data'
csv_list = []

csv_to_read = 15
file_count = 0
for csv in os.listdir(src_dir_path):
    if csv.endswith(".csv"):
        print(f'read next: {csv}')
        csv_path = os.path.join(src_dir_path, csv)
        df = pd.read_csv(csv_path)

        # delete isFallen == 1, bc Robot doesn't need to check for falling while falling
        timestamps_to_use = df[df['isFallen'] != 1]

        # Filter data to 50/50 for will fall in 2s and stable walking (against Overfitting)
        pos_sets = timestamps_to_use[timestamps_to_use['willFall_dt'] <= 2 * 10 ** 6]
        neg_sets = timestamps_to_use[timestamps_to_use['willFall_dt'] > 2 * 10 ** 6]
        to_remove = len(neg_sets) - len(pos_sets)
        neg_sets = neg_sets.drop(np.random.choice(neg_sets.index, to_remove, replace=False))
        timestamps_to_use = pd.concat([pos_sets['timestamp'], neg_sets['timestamp']])

        # Menge reduzieren für Testfälle/Debug
        #timestamps_to_use = timestamps_to_use.drop(np.random.choice(timestamps_to_use.index, round(len(timestamps_to_use)*.98), replace=False))

        print(f'Data set length: {len(timestamps_to_use)}')

        # timestamps_to_use.head()
        # timestamps_to_use.info()
        # print(timestamps_to_use)

        # Shape inputs to (200,6)
        time_steps = 200
        inputs = pd.DataFrame(columns=['vals', 'willFall_dt'])
        old_p = .0
        count = 0
        for i in timestamps_to_use.index:
            if i < time_steps - 1:
                continue  # skip erste Zeilen, die nicht genügend Vorgänger-Einträge haben

            vals_i = pd.DataFrame(columns=['gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ'])
            for j in range(time_steps):
                vals_ij = df.iloc[i - j][['gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ']]
                vals_i.loc[len(vals_i)] = vals_ij

            entry = pd.Series({'vals': vals_i, 'willFall_dt': df.iloc[i]['willFall_dt']})
            inputs.loc[len(inputs)] = entry

            # Show progress per CSV
            count += 1
            p = round(count / len(timestamps_to_use) * 100, 1)
            if p > old_p:
                print(f'progress {p} %', end='\r')
                old_p = p

        inputs['willFall'] = inputs.apply(lambda row:
                                          0 if row['willFall_dt'] > 2 * (10 ** 6)
                                          else 1 if row['willFall_dt'] > 1 * (10 ** 6)
                                          else 2 if row['willFall_dt'] > .5 * (10 ** 6)
                                          else 3 if row['willFall_dt'] > .2 * (10 ** 6)
                                          else 4, axis=1)

        csv_list.append(inputs)

        file_count += 1
        if file_count > csv_to_read - 1:
            break

df = pd.concat(csv_list, axis=0, ignore_index=True)

# Constructing the input
print()
X = df['vals']

# Constructing the output
y = df['willFall']
num_classes = len(np.unique(y))
#encoder = LabelEncoder()
#encoder.fit(y)
#enc_y = encoder.transform(y)
#dum_y = to_categorical(y, num_classes=num_classes)

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True)

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
def make_FCN_model(input_shape):
    """
    Defining the model
    :param input_shape: shape of inputs
    :return: FCN model
    """
    input_layer = keras.layers.Flatten(input_shape=input_shape)

    hidden_layer = keras.layers.Dense(256, activation='relu')(input_layer)

    output_layer = keras.layers.Dense(num_classes, activation='softmax')(hidden_layer)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


# Defining Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.keras', save_best_only=True, monitor='val_loss'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, verbose=1
    )
]
# Defining the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Fitting the model
history = model.fit(
    X_train_tensor,
    y_train,
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1)

# Evaluating the model
print('MODEL EVALUATION')
model = keras.models.load_model('best_model.keras')
# Model Evaluation
test_loss, test_acc = model.evaluate(X_test_tensor, y_test)#batchsize = 8

print('Test accuracy', test_acc)
print('Test loss', test_loss)

# Model summary
print('MODEL SUMMARY')
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

# Confusion matrix
print('CONFUSION MATRIX')
#y_pred = model.predict(X_test_tensor)
#encoder = LabelEncoder()
#encoder.fit(y_test)
#enc_y = encoder.transform(y_test)
#dum_y = to_categorical(y_test, num_classes=num_classes)
#print(confusion_matrix(dum_y, y_pred))

# model.save('./Models/bigInputANN/bigInput.keras')
