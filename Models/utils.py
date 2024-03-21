import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import numpy as np
import matplotlib.pyplot as plt


def get_flops(model):
    """
    Calculates FLOPs of a given model
    :param model: model
    :return: Number of FLOPs
    """
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


def get_fpr_per_class_srnn(model, x, y):
    """
    !!NICHT MIT IN ANGABE CODE REINNEHMEN!!
    Calculates the false-positive-rate for each class for a stateful RNN model on a given test data set
    :param model: model
    :param x: input test data
    :param y: output test data
    :return: fpr for each class
    """
    pred_counts = {
        0: {'total': 0,
            'fp': 0,
            'fn': 0},
        1: {'total': 0,
            'fp': 0,
            'fn': 0},
        2: {'total': 0,
            'fp': 0,
            'fn': 0},
        3: {'total': 0,
            'fp': 0,
            'fn': 0},
        4: {'total': 0,
            'fp': 0,
            'fn': 0},
        5: {'total': 0,
            'fp': 0,
            'fn': 0}
    }

    truths = y['willFall'].values

    model.reset_states()


    # plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Klassenflanken und FP/FN ermitteln
    ts_flanks = [0]
    current_class_space = y.iloc[0]['willFall']
    fp_class0 = []
    count = 0
    put_first_label = True
    for (truth, x_i) in zip(y['willFall'].values, x):
        print(f'{count}/{len(y)}', end='\r')
        # Flanken ermitteln
        if current_class_space == 0 and truth == 1:
            ts_flanks.append(count)
            current_class_space = 1
        elif current_class_space == 1 and truth == 0:
            ts_flanks.append(count)
            current_class_space = 0

        # FP/FN ermitteln
        x_rs = np.reshape(x_i, (1, 1, 6))
        prediction = np.argmax(model.predict_on_batch(x_rs), axis=1)
        pred_counts[truth]['total'] += 1
        if prediction != truth:
            if prediction == 0:
                if put_first_label:
                    ax.stem(count, 2, linefmt='red', label='False Positve Sturzvorhersage')  # FP für Klasse 0 rot einfärben
                    put_first_label = False
                else:
                    ax.stem(count, 2, linefmt='red')
            pred_counts[prediction[0]]['fp'] += 1
            pred_counts[truth]['fn'] += 1

        count += 1
    ts_flanks.append(len(y))

    color = 'blue'
    for i in range(len(ts_flanks)-1):
        x = [ts_flanks[i], ts_flanks[i+1]]
        ax.stackplot(x, [-2, -2], colors=color)
        if color == 'blue':
            color = 'green'
        else:
            color = 'blue'

    ax.set_yticks([])
    plt.xlabel('Zeitstempel alle 12 Millisekunden')
    plt.title('Klassifizierte Daten entlang des Zeitstrahls')
    plt.legend()
    plt.show()

    return pred_counts
