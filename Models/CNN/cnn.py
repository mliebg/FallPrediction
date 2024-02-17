import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

###################################################
# PREPROCESSING just like Fully Connected Network
###################################################
# Import Data
src_dir_path = './DataAcquisition/DataLabeling/fullyLabeled/data'
csv_list = []

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
        if file_count > 15:
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

