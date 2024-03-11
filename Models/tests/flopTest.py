import keras
import tensorflow as tf
# 256x256x160 -> 520864
# 128x128x128 -> 220830
model = keras.Sequential([
    # FCN
    #keras.layers.Conv1D(filters=26, kernel_size=4, input_shape=(100, 6)),
    #keras.layers.MaxPooling1D(2),
    #keras.layers.Conv1D(filters=26, kernel_size=4),
    #keras.layers.MaxPooling1D(2),
    #keras.layers.Conv1D(filters=26, kernel_size=4),
    #keras.layers.MaxPooling1D(2),
    # RNN
    keras.layers.LSTM(20, input_shape=(100, 6)),

    # MLP
    #keras.layers.Flatten(input_shape=(200, 6)),
    #keras.layers.Dense(160, activation='sigmoid'),
    #keras.layers.Dense(160, activation='sigmoid'),
    #keras.layers.Dense(160, activation='sigmoid'),
    keras.layers.Dense(5, activation='softmax')
])

model.summary()

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


print(f"FLOPs: {get_flops(model)*100}")