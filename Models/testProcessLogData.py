import keras
import tensorflow as tf
#import tensorrt
import keras.utils
import numpy as np
import pandas as pd
import queue


# mean:
mean_gyroYaw   = -3.371571e-03
mean_gyroPitch = 1.453301e-03
mean_gyroRoll  = 2.135230e-03
mean_accelX    = 9.700241e-01
mean_accelY    = 5.375635e-02
mean_accelZ    = -9.269141e+00
# std: 
std_gyroYaw   = 7.106031e-01
std_gyroPitch = 4.883810e-01
std_gyroRoll  = 4.130802e-01
std_accelX    = 2.788253e+00
std_accelY    = 2.105054e+00
std_accelZ    = 3.450007e+00

print('TEST RUN PROCESS A LOG FILE')
logfile_path = '/home/dev/repos/FallPrediction/DataAcquisition/DataLabeling/FullyLabeled/data/excludeForTests/imu_5pathfinder-11-25-34_labeled.csv'
timesteps = 50

# load test log data
log_ds = pd.read_csv(logfile_path)
len_log = len(log_ds)

# normalize data with z-norm params
log_ds['gyroYaw'] = (log_ds['gyroYaw'] - mean_gyroYaw) / std_gyroYaw
log_ds['gyroPitch'] = (log_ds['gyroPitch'] - mean_gyroPitch) / std_gyroPitch
log_ds['gyroRoll'] = (log_ds['gyroRoll'] - mean_gyroRoll) / std_gyroRoll
log_ds['accelX'] = (log_ds['accelX'] - mean_accelX) / std_accelX
log_ds['accelY'] = (log_ds['accelY'] - mean_accelY) / std_accelY
log_ds['accelZ'] = (log_ds['accelZ'] - mean_accelZ) / std_accelZ

# load model
model = keras.models.load_model(filepath='/home/dev/repos/FallPrediction/Models/Evaluation/keras-models/best_lstm50.keras')
#keras.utils.plot_model(model,show_shapes=True)

# simulate data coming in every 12 ms
# fill input data X as the 50 last incoming IMU data
vals_q = queue.Queue()   # np.empty((50,6))
fn_i = []
fp_i = []
correct_count = 0
for index, row in log_ds.iterrows():
    if index % 100 == 0:
         print(f'progress: {index}/{len_log}',end='\r')
    vals_q.put([row['gyroYaw'],row['gyroPitch'],row['gyroRoll'],row['accelX'],row['accelY'],row['accelZ']])
    if index >= timesteps -1:
        # skip isFallen = 1
        if row['isFallen'] == 1:
            vals_q.get()
            continue

        ql = list(vals_q.queue)
        ql.reverse()
        X_np = np.array(ql)
        #print(X_np.shape)
        X_resh = np.reshape(X_np, (1, 50, 6))

        truth = 2 * 10 ** 6 >= row['willFall_dt']
        pred_val = model.predict(X_resh, verbose=0)
        pred = pred_val > 0.5

        #print(f'willFall in 2s at Index={index}\nprediction : {pred} ({pred_val}) | truth : {truth}')
        if pred == truth:
            #print('correctly predicted')
            correct_count += 1
        elif pred == True:            
            #print('wrongly predicted')
            fp_i.append(index)
        elif pred == False:
            fn_i.append(index)

        vals_q.get()
        #input()

# evaluate results
count_isFallen = (log_ds['isFallen'] == 1).sum()
total = len_log - timesteps - count_isFallen
acc = correct_count / total
fp = len(fp_i)
fn = len(fn_i)

# mean distance to True Positive MDTP
sum = 0
for i in fp_i:
    sum += log_ds.iloc[i]['willFall_dt'] - 2 * 10 ** 6
MDTP = sum / len(fp_i)

print(f'Log data from {logfile_path} was processed with a total of {total} values ({len_log} minus {timesteps} to fill value queue for the first time minus {count_isFallen} where isFallen = 1).\n  Accuracy: {acc}\n  False Pos: {fp}\n  False Neg: {fn}\n  MDTP : {MDTP} in mys ')

'''
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
'''