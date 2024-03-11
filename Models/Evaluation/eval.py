import keras
import pickle
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import os
import numpy as np

def get_afpr(model, x_tensor, y):
    y_predict = model.predict(x_tensor)
    y_predicts = np.argmax(y_predict, axis=1)
    i = 0
    total_neg = 0
    false_pos = 0
    for y_i in y:
        if y_predicts[i] > 0:
            pred = 1
        else:
            pred = 0

        if y_i == 0:
            truth = 0
            total_neg += 1
            if truth != pred:
                false_pos += 1
        i += 1
    return round(false_pos / total_neg, 4)


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


tss = [100, 200]  # type of inputs (length of time series)
models_dir_path = './Models/Evaluation/models'  # directory of saved models

for ts in tss:
    # Load Test data
    print('loading test data')
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

    # Search for models
    models = []
    for file in os.listdir(models_dir_path):
        if file.endswith(f'{ts}.keras'):
            models.append(os.path.join(models_dir_path, file))

    for model_path in models:
        # Evaluating the model
        print('MODEL EVALUATION')
        print(f'\n\nnext model: {model_path}')
        model = keras.models.load_model(filepath=model_path)

        # Model summary
        print('MODEL SUMMARY')
        model.summary()

        # Model Evaluation
        test_loss, test_acc = model.evaluate(X_test_tensor, y_test)
        test_afpr = get_afpr(model, X_test_tensor, y_test)
        print('Test AFPR: ', test_afpr)
        #print('Test accuracy', test_acc)
        #print('Test loss', test_loss)

        flops = get_flops(model)
        print(f"FLOPs: {flops * ts} ({flops})")
