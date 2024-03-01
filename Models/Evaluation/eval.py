import keras
import pickle
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import os


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


# Program Parameters
ts = 200  # type of inputs (length of time series)
models_dir_path = ''  # directory of saved models

# Load Test data
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
for keras in os.listdir(models_dir_path):
    if keras.endswith(f'{ts}.keras'):
        keras_path = os.path.join(models_dir_path, keras)
        models.add(keras_path)


for model_path in models:
    # Evaluating the model
    print('MODEL EVALUATION')
    print(f'next model: {model_path}')
    model = keras.models.load_model(model_path)

    # Model Evaluation
    test_loss, test_acc = model.evaluate(X_test_tensor, y_test)

    print('Test accuracy', test_acc)
    print('Test loss', test_loss)

    # Model summary
    print('MODEL SUMMARY')
    model.summary()

    print(f"FLOPs: {get_flops(model)}")
