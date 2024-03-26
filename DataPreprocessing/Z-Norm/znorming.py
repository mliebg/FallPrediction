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
        df.to_csv(f'./DataAcquisition/DataLabeling/StatefulRnnData/standardized_data/data/{out_csv}', columns=['gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ', 'isFallen', 'willFall_dt'], index_label='ts')

print(f'mean: {mean}\nstd: {std}')
