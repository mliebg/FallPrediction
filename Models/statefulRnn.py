# https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
# Stateful LSTM to learn one-char to one-char mapping
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os
import pickle
import keras
from utils import get_flops, get_fpr_per_class_srnn

# fix random seed for reproducibility
tf.random.set_seed(42)

for units_num in [64, 128, 256]:
    # Model Definition
    batch_size = 1
    model = Sequential()
    model.add(LSTM(units_num, batch_input_shape=(batch_size, 1, 6), stateful=True, dropout=0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    # todo: fix get_flops() with stateful RNN and input shape; workaround:
    modelF = Sequential()
    modelF.add(LSTM(units_num, input_shape=(1, 6), dropout=0.5))
    modelF.add(Dense(6, activation='softmax'))
    modelF.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    print(f'FLOPs (without stateful): {get_flops(modelF)}')
    model.summary()

    # Model Training
    print('train model')
    src_dir_path = './DataAcquisition/DataLabeling/StatefulRnnData/labeled/data'
    for _ in range(2):
        for pkl in os.listdir(src_dir_path):
            if pkl.endswith(".pkl"):
                # load train data
                print(f'next training run with: {pkl}')
                with open(os.path.join(src_dir_path, pkl), 'rb') as f:
                    train_data = pickle.load(f)
                f.close()
                X = train_data.iloc[:, :-1]
                y = train_data.iloc[:, -1:]

                # reshape X to be [samples, time steps, features]
                X = np.reshape(X, (len(X), 1, 6))

                model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
                model.reset_states()
                break
            break

    model.save(f'./Models/units{units_num}-stflRnn.keras')

    # Model Evaluation
    print(f'Evaluate LSTM ({units_num})\nloading test data')
    with open('./DataAcquisition/DataLabeling/StatefulRnnData/labeled/data/test/test_imu_4cometinterceptor-11-25-34_testdata.csv.pkl', 'rb') as f:
        test_data = pickle.load(f)
    f.close()
    X = test_data.iloc[:, :-1]
    X = np.reshape(X, (len(X), 1, 6))
    y = test_data.iloc[:, -1:]

    model = keras.models.load_model(f'./Models/units{units_num}-stflRnn.keras')

    # SCA overall
    model.evaluate(X, y, batch_size=1)

    # FPR for each class
    fpr_per_class = get_fpr_per_class_srnn(model, X, y)
    print(f'FPR per class:\n'
          f'(0) stable walking  : {fpr_per_class[0]}\n'
          f'(1) falling in 2.0s : {fpr_per_class[1]}\n'
          f'(2) falling in 1.0s : {fpr_per_class[2]}\n'
          f'(3) falling in 0.5s : {fpr_per_class[3]}\n'
          f'(4) falling in 0.2s : {fpr_per_class[4]}\n'
          f'(5) fallen/dont care: {fpr_per_class[5]}')
