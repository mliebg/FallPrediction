import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import tempfile
import os

# Hyperparameter
logRes_threshold = 0.5
epochs = 200
learning_rate = 0.0001

print(tf.__version__)
# To make the results reproducible, set the random seed value.
tf.random.set_seed(22)

# Importing the data
dir_path = './DataAcquisition/DataLabeling/fullyLabeled/data'
csv_list = []
my_sec_dot5 = 1 * 10 ** 6

for csv in os.listdir(dir_path):
    if csv.endswith(".csv"):
        csv_path = os.path.join(dir_path, csv)
        dataframe = pd.read_csv(csv_path)
        # isFallen = 1 kann entfernt werden, da der Roboter während des Sturzes/Aufstehens nicht prüfen muss, ob er fällt
        dataframe = dataframe[dataframe['isFallen'] != 1]
        dataframe['willFall_in2s'] = dataframe.apply(lambda row: 1 if row['willFall_dt'] <= my_sec_dot5 else 0, axis=1)
        csv_list.append(dataframe)

dataset = pd.concat(csv_list, axis=0, ignore_index=True)
# 50 / 5ß ratio für willFall_in2s, um overfitting zu vermeiden
pos_sets = dataset[dataset['willFall_in2s'] == 1]
neg_sets = dataset[dataset['willFall_in2s'] == 0]
to_remove = len(neg_sets) - len(pos_sets)
neg_sets = neg_sets.drop(np.random.choice(neg_sets.index, to_remove, replace=False))
dataset = pd.concat([pos_sets, neg_sets], axis=0, ignore_index=True)
dataset.info()
dataset.head()

# Split data
train_dataset = dataset.sample(frac=0.7, random_state=1)
test_dataset = dataset.drop(train_dataset.index)
#validation_dataset = dataset.drop(train_dataset.index)
#test_dataset = validation_dataset.sample(frac=0.33, random_state=1)
#validation_dataset = validation_dataset.drop(test_dataset.index)
print(len(train_dataset))
#print(len(validation_dataset))
print(len(test_dataset))
#input()

# The `id` column can be dropped since each row is unique
x_train, y_train = train_dataset.iloc[:, 1:6], train_dataset.iloc[:, 9]
#x_valid, y_valid = validation_dataset.iloc[:, 1:6], validation_dataset.iloc[:, 9]
x_test, y_test = test_dataset.iloc[:, 1:6], test_dataset.iloc[:, 9]

x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
#x_valid, y_valid = tf.convert_to_tensor(x_valid, dtype=tf.float32), tf.convert_to_tensor(y_valid, dtype=tf.float32)
x_test, y_test = tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)


#train_dataset.describe().transpose()[:10]


class Normalize(tf.Module):
    def __init__(self, x):
        # Initialize the mean and standard deviation for normalization
        self.mean = tf.Variable(tf.math.reduce_mean(x, axis=0))
        self.std = tf.Variable(tf.math.reduce_std(x, axis=0))

    def norm(self, x):
        # Normalize the input
        return (x - self.mean)/self.std

    def unnorm(self, x):
        # Unnormalize the input
        return (x * self.std) + self.mean


norm_x = Normalize(x_train)
x_train_norm, x_test_norm = norm_x.norm(x_train), norm_x.norm(x_test)

def log_loss(y_pred, y):
    # Compute the log loss function
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(ce)


class LogisticRegression(tf.Module):

    def __init__(self):
        self.built = False

    def __call__(self, x, train=True):
        # Initialize the model parameters on the first call
        if not self.built:
            # Randomly generate the weights and the bias term
            rand_w = tf.random.uniform(shape=[x.shape[-1], 1], seed=22)
            rand_b = tf.random.uniform(shape=[], seed=22)
            self.w = tf.Variable(rand_w)
            self.b = tf.Variable(rand_b)
            self.built = True
        # Compute the model output
        z = tf.add(tf.matmul(x, self.w), self.b)
        z = tf.squeeze(z, axis=1)
        if train:
            return z
        return tf.sigmoid(z)


log_reg = LogisticRegression()
y_pred = log_reg(x_train_norm, train=False)
y_pred.numpy()


def predict_class(y_pred, thresh=logRes_threshold):
    # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
    return tf.cast(y_pred > thresh, tf.float32)


def accuracy(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`
    y_pred = tf.math.sigmoid(y_pred)
    y_pred_class = predict_class(y_pred)
    check_equal = tf.cast(y_pred_class == y, tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val


batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train))
train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test))
test_dataset = test_dataset.shuffle(buffer_size=x_test.shape[0]).batch(batch_size)

# Set training parameters
train_losses, test_losses = [], []
train_accs, test_accs = [], []

# Set up the training loop and begin training
for epoch in range(epochs):
    batch_losses_train, batch_accs_train = [], []
    batch_losses_test, batch_accs_test = [], []

    # Iterate over the training data
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            y_pred_batch = log_reg(x_batch)
            batch_loss = log_loss(y_pred_batch, y_batch)
        batch_acc = accuracy(y_pred_batch, y_batch)
        # Update the parameters with respect to the gradient calculations
        grads = tape.gradient(batch_loss, log_reg.variables)
        for g,v in zip(grads, log_reg.variables):
            v.assign_sub(learning_rate * g)
        # Keep track of batch-level training performance
        batch_losses_train.append(batch_loss)
        batch_accs_train.append(batch_acc)

    # Iterate over the testing data
    for x_batch, y_batch in test_dataset:
        y_pred_batch = log_reg(x_batch)
        batch_loss = log_loss(y_pred_batch, y_batch)
        batch_acc = accuracy(y_pred_batch, y_batch)
        # Keep track of batch-level testing performance
        batch_losses_test.append(batch_loss)
        batch_accs_test.append(batch_acc)

    # Keep track of epoch-level model performance
    train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
    test_loss, test_acc = tf.reduce_mean(batch_losses_test), tf.reduce_mean(batch_accs_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Training log loss: {train_loss:.3f}")

# Evaluation
plt.plot(range(epochs), train_losses, label = "Training loss")
plt.plot(range(epochs), test_losses, label = "Testing loss")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.legend()
plt.title("Log loss vs training iterations")

plt.plot(range(epochs), train_accs, label = "Training accuracy")
plt.plot(range(epochs), test_accs, label = "Testing accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy vs training iterations")

print(f"Final training log loss: {train_losses[-1]:.3f}")
print(f"Final testing log Loss: {test_losses[-1]:.3f}")

print(f"Final training accuracy: {train_accs[-1]:.3f}")
print(f"Final testing accuracy: {test_accs[-1]:.3f}")
print(f"Hyperparameter:\nActivation threshold: {logRes_threshold}\nEpochs: {epochs}\nLearning rate: {learning_rate}")



def show_confusion_matrix(y, y_classes, typ):
    # Compute the confusion matrix and normalize it
    plt.figure(figsize=(10, 10))
    confusion = sk_metrics.confusion_matrix(y.numpy(), y_classes.numpy())
    confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)
    axis_labels = range(2)
    ax = sns.heatmap(
        confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
        cmap='Blues', annot=True, fmt='.4f', square=True)
    plt.title(f"Confusion matrix: {typ}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


y_pred_train, y_pred_test = log_reg(x_train_norm, train=False), log_reg(x_test_norm, train=False)
train_classes, test_classes = predict_class(y_pred_train), predict_class(y_pred_test)

show_confusion_matrix(y_train, train_classes, 'Training')

show_confusion_matrix(y_test, test_classes, 'Testing')
plt.show();
# Modell speichern
'''
class ExportModule(tf.Module):
    def __init__(self, model, norm_x, class_pred):
        # Initialize pre- and post-processing functions
        self.model = model
        self.norm_x = norm_x
        self.class_pred = class_pred

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def __call__(self, x):
        # Run the `ExportModule` for new data points
        x = self.norm_x.norm(x)
        y = self.model(x, train=False)
        y = self.class_pred(y)
        return y


log_reg_export = ExportModule(model=log_reg,
                              norm_x=norm_x,
                              class_pred=predict_class)

models = tempfile.mkdtemp()
save_path = './Models/log_reg_export'
tf.saved_model.save(log_reg_export, save_path)
print(f"saved to {save_path}")
log_reg_loaded = tf.saved_model.load(save_path)
test_preds = log_reg_loaded(x_test)
test_preds[:10].numpy()
'''