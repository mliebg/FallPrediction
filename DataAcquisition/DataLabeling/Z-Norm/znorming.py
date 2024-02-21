import pandas as pd
import os

src_dir_path = './DataAcquisition/DataLabeling/FullyLabeled/data/'

# calc total mean and std over all files
print('calculating mean and std over all data')
dfs = []
for csv in os.listdir(src_dir_path):
    if csv.endswith(".csv"):
        csv_path = os.path.join(src_dir_path, csv)
        df = pd.read_csv(csv_path)
        dfs.append(df)

df_total = pd.concat(dfs, ignore_index=True)
mean = df_total.mean()
std = df_total.std()
# apply z-norm to every csv and save new data
for csv in os.listdir(src_dir_path):
    if csv.endswith(".csv"):
        print(f'read next: {csv}')

        csv_path = os.path.join(src_dir_path, csv)
        df = pd.read_csv(csv_path)

        # Z-Normalization
        # z = (x - my) / sig
        df['gyroYaw'] = (df['gyroYaw'] - mean['gyroYaw']) / std['gyroYaw']
        df['gyroPitch'] = (df['gyroPitch'] - mean['gyroPitch']) / std['gyroPitch']
        df['gyroRoll'] = (df['gyroRoll'] - mean['gyroRoll']) / std['gyroRoll']
        df['accelX'] = (df['accelX'] - mean['accelX']) / std['accelX']
        df['accelY'] = (df['accelY'] - mean['accelY']) / std['accelY']
        df['accelZ'] = (df['accelZ'] - mean['accelZ']) / std['accelZ']

        out_csv = csv[:-11] + 'z-norm.csv'
        df.to_csv(f'./DataAcquisition/DataLabeling/Z-Norm/data/{out_csv}', columns=['gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ', 'isFallen', 'willFall_dt'], index_label='ts')

print(f'mean: {mean}\nstd: {std}')
'''
mean:
gyroYaw       -2.507752e-03
gyroPitch      1.595531e-03
gyroRoll       1.680665e-03
accelX         9.988282e-01
accelY         5.180147e-02
accelZ        -9.264773e+00
dtype: float64

std:
gyroYaw        6.969683e-01
gyroPitch      4.822158e-01
gyroRoll       4.113075e-01
accelX         2.747201e+00
accelY         2.094108e+00
accelZ         3.409107e+00
dtype: float64
'''