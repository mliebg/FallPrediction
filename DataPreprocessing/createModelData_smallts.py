import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Import Data
src_dir_path = './DataPreprocessing/Z-Norm/data'
csv_list = []

# Go through all CSV
total_data = 0
for csv in os.listdir(src_dir_path):
    if csv.endswith(".csv"):
        print(f'read next: {csv}')
        csv_path = os.path.join(src_dir_path, csv)
        df = pd.read_csv(csv_path)


        # Remove isFallen == 1, bc Robot doesn't need to check for falling while falling
        timestamps_to_use = df[df['isFallen'] != 1]


        # Filter data to same amounts per class (against Overfitting)
        df_stable = timestamps_to_use[timestamps_to_use['willFall_dt'] > 2 * 10 ** 6]
        df_2 = timestamps_to_use[2 * 10 ** 6 >= timestamps_to_use['willFall_dt']]

        df_stable = df_stable.drop(np.random.choice(df_stable.index, len(df_stable) - len(df_2), replace=False))

        timestamps_to_use = pd.concat([df_stable['ts'], df_2['ts']])

        print(f'total: {len(timestamps_to_use)}   '
              f'0: {len(df_stable)}   '
              f'1: {len(df_2)}   ')

        print(f'Data set length: {len(timestamps_to_use)}')
        total_data += len(timestamps_to_use)

        time_steps = 50
        inputs = pd.DataFrame(columns=['vals', 'willFall_dt'])
        old_p = .0
        count = 0
        for i in timestamps_to_use.index:
            if i < time_steps - 1:
                continue  # skip first rows that have not enough rows before them to form a sequence

            vals_i = pd.DataFrame(columns=['gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ'])
            for j in range(time_steps):
                vals_ij = df.iloc[i - j][['gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ']]
                vals_i.loc[len(vals_i)] = vals_ij

            entry = pd.Series({'vals': vals_i, 'willFall_dt': df.iloc[i]['willFall_dt']})
            inputs.loc[len(inputs)] = entry

            # Show progress per CSV
            count += 1
            p = round(count / len(timestamps_to_use) * 100, 1)
            if p > old_p:
                print(f'progress {p} %', end='\r')
                old_p = p

        inputs['willFallin2'] = inputs.apply(lambda row: 0 if row['willFall_dt'] > 2 * (10 ** 6) else 1, axis=1)

        csv_list.append(inputs)


print(f'total data amount: {total_data}')  # 19875 without Goalie
df = pd.concat(csv_list, axis=0, ignore_index=True)

# Constructing the input
X = df['vals']

# Constructing the output
y = df['willFallin2']

# Split the data into train/vali/test set ~(70/15/15)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.176, random_state=42, shuffle=True)


# Dump data for later use in model training
with open(f"ts{time_steps}_xtrain.pkl", "wb") as f:
    pickle.dump(X_train, f)
f.close()
with open(f"ts{time_steps}_xval.pkl", "wb") as f:
    pickle.dump(X_val, f)
f.close()
with open(f"ts{time_steps}_xtest.pkl", "wb") as f:
    pickle.dump(X_test, f)
f.close()
with open(f"ts{time_steps}_ytrain.pkl", "wb") as f:
    pickle.dump(y_train, f)
f.close()
with open(f"ts{time_steps}_yval.pkl", "wb") as f:
    pickle.dump(y_val, f)
f.close()
with open(f"ts{time_steps}_ytest.pkl", "wb") as f:
    pickle.dump(y_test, f)
f.close()
