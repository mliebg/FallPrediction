import numpy as np
from utils import print_stamped_message 
import argparse
import json
import os
import pickle
import tensorflow as tf
import tensorrt
import keras
import matplotlib.pyplot as plt
from keras import layers
from main import get_config_file
#from keras.layers import CuDNNLSTM

##########################################
# Long Short-Term Memory only 1 class ##///and 50 timestamps per time series
##########################################

def main():
    script_chaining = False
    print_stamped_message(2, f"GPUs available: {tf.config.list_physical_devices('GPU')}")

    parser = argparse.ArgumentParser(description="Read parameters from a JSON config file.")
    parser.add_argument("log_file", type=str, help="Path to the log file")    
    args = parser.parse_args()

    # Read the config file
    config = get_config_file()

    # Load parameter from config
    model_data_src = config["build"]["data_dir"]  # directory of model data (pickle)
    lstm_units = config["build"]["lstm_units"]    # number of units used in lstm layer
    ts = config['data']['time_steps']             # number of timesteps that are used in input range
    range = config['data']['trigger_range']       # time before a fall that classifies as willFall later in model (in seconds)
    dropout = 0.0
    

    if script_chaining:
        print_stamped_message(2,f'BEGIN BUILDING AND TRAINING LSTM\n Parameter Settings:\n  source dir:    {model_data_src}\n  lstm units:    {lstm_units}')
    else:
        print_stamped_message(2,f'BEGIN BUILDING AND TRAINING LSTM\n Parameter Settings:\n  source dir:    {model_data_src}\n  lstm units:    {lstm_units}\n  time steps:    {ts}\n  trigger range: {range}\n  LOG output: {args.log_file}')


    # Load Data
    print_stamped_message(2,'load model data')
    
    # Constructing the input
    with open(os.path.join(model_data_src, f'ts{ts}_range{range}_xtrain.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    f.close()
    with open(os.path.join(model_data_src, f'ts{ts}_range{range}_xval.pkl'), 'rb') as f:
        X_val = pickle.load(f)
    f.close()
    # Constructing the output
    with open(os.path.join(model_data_src, f'ts{ts}_range{range}_ytrain.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    f.close()
    with open(os.path.join(model_data_src, f'ts{ts}_range{range}_yval.pkl'), 'rb') as f:
        y_val = pickle.load(f)
    f.close()

    num_labels = len(np.unique(y_train))

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
        model = keras.models.Sequential()
        model.add(layers.LSTM(lstm_units, input_shape=input_shape,dropout=dropout))
        if num_labels == 2:
            acti = 'relu'
        else:
            acti = 'softmax'

        print_stamped_message(2, f'found {num_labels} labels in data set --> using {acti} activation')
        model.add(layers.Dense(1, activation=acti))

        # Compile
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                    loss='binary_crossentropy',
                    metrics=['binary_accuracy'])

        return model


    model = build_model()

    script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_parent_dir, f"Code/Models/keras-models/lstm{lstm_units}-{ts}-{range}.keras")
    
    # Defining Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path, save_best_only=True, monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=20, min_lr=0.0000001
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50, verbose=1
        )
    ]

    history = model.fit(
        X_train_tensor,
        y_train,
        epochs=1000,
        batch_size=24,
        callbacks=callbacks,
        validation_data=(X_val_tensor, y_val),
        verbose=1)
    print_stamped_message(2, f"Best LSTM will be saved as {model_path}")

    # Plot model loss
    metric = 'binary_accuracy'
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('LSTM Training' + metric)
    plt.ylabel('binary accuracy (BA)', fontsize='large')
    plt.xlabel('Epoche', fontsize='large')
    plt.legend(['Trainingsset BA', 'Validierungsset BA'], loc='best')
    plt.show()


if __name__ == "__main__":
    main()
