import keras
import pickle
import tensorflow as tf
#import tensorrt
import keras.utils
from utils import get_flops
import os
import numpy as np


def get_fp_and_fn(model, x, y):
    """
    Counts false-positives/false-negatives for a model on a test data set.
    Also counts total positives of each class
    :param model: model
    :param x: input test data
    :param y: output test data
    :return: total p/fp/fn for each class
    """
    pred_counts = {
        0: {'p': 0,
            'fp': 0,
            'fn': 0},
        1: {'p': 0,
            'fp': 0,
            'fn': 0},
        2: {'p': 0,
            'fp': 0,
            'fn': 0},
        3: {'p': 0,
            'fp': 0,
            'fn': 0},
        4: {'p': 0,
            'fp': 0,
            'fn': 0}
    }

    count = 1
    y_predict = np.argmax(model.predict(x), axis=1)
    for (truth, predict) in zip(y.values, y_predict):
        print(f'{count}/{len(y)}', end='\r')

        # P/FP/FN ermitteln
        pred_counts[truth]['p'] += 1
        if predict != truth:
            pred_counts[predict]['fp'] += 1
            pred_counts[truth]['fn'] += 1

        count += 1

    return pred_counts


print('MODEL EVALUATION')
models_dir_path = './Models/Evaluation/keras-models'  # directory of saved keras-models

for ts in [50]:  # type of inputs (length of time series) 100, 200
    # Load Test data
    print(f'loading test data for sequence len: {ts}')
    # Inputs
    with open(f'./DataPreprocessing/ts{ts}/ts{ts}_xtest.pkl', 'rb') as f:
        X_test = pickle.load(f)
    f.close()
    arrays = [df.values for df in X_test]
    stacked = tf.stack(arrays)
    X_test_tensor = tf.convert_to_tensor(stacked)
    # Outputs
    with open(f'./DataPreprocessing/ts{ts}/ts{ts}_ytest.pkl', 'rb') as f:
        y_test = pickle.load(f)
    f.close()

    # Search for keras-models
    models = []
    for file in os.listdir(models_dir_path):
        if file.endswith(f'{ts}.keras'):
            models.append(os.path.join(models_dir_path, file))

    for model_path in models:
        print(f'\n\nnext model: {model_path}')
        model = keras.models.load_model(filepath=model_path)

        # Model Summary
        print('MODEL SUMMARY')
        model.summary()
        keras.utils.plot_model(model)

        # Model Evaluation
        test_loss, test_acc = model.evaluate(X_test_tensor, y_test)
        #fp_fn = get_fp_and_fn(model, X_test_tensor, y_test)
        #print(f'Test FPs/FNs:\n'
        #      f'(0) stable walking  : {fp_fn[0]}\n'
        #      f'(1) falling in 2.0s : {fp_fn[1]}\n'
        #      f'(2) falling in 1.0s : {fp_fn[2]}\n'
        #      f'(3) falling in 0.5s : {fp_fn[3]}\n'
        #      f'(4) falling in 0.2s : {fp_fn[4]}\n')

        print('Test accuracy', test_acc)
        print('Test loss', test_loss)

        flops = get_flops(model)
        print(f'FLOPs: {flops}')
