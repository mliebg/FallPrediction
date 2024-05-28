import os
import json
import argparse
import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def print_stamped_message(level, msg):
    prefix = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match level: 
        case 0: prefix += ' TRACE : '
        case 1: prefix += ' DEBUG : '
        case 2: prefix += '  INFO : '
        case 3: prefix += ' ERROR : '
        case _: prefix += '       : '
    
    print(prefix + msg)


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description="Read parameters from a JSON config file.")
    parser.add_argument('config_file', type=str, help="Path to the JSON config file")
    
    args = parser.parse_args()

    # Read the config file
    config = read_config(args.config_file)

    # Load parameter from config
    src_dir_path = config.get('input_dir')        # directory of sensor data (CSVs)
    out_dir_path = config.get('output_dir')       # output directory of model data (pickle)
    time_steps = config.get('time_steps')         # number of timesteps that are used in input range
    trigger_range = config.get('trigger_range')   # time before a fall that classifies as willFall later in model (in seconds)

    new_dir = f'ts{time_steps}_range{trigger_range}_data'

    print_stamped_message(2, f'BEGIN CREATING MODEL DATA\n PARAMETERS:\n  input dir:     {src_dir_path}\n  output dir:    {out_dir_path}\n  time steps:    {time_steps}\n  trigger range: {trigger_range} seconds')
    
    # Go through all CSV
    total_data = 0
    csv_list = []
    for csv in os.listdir(src_dir_path):
        if csv.endswith(".csv"):
            print_stamped_message(2, f'read next: {csv}')
            csv_path = os.path.join(src_dir_path, csv)
            df = pd.read_csv(csv_path)


            # Remove isFallen == 1, bc Robot doesn't need to check for falling while falling
            timestamps_to_use = df[df['isFallen'] != 1]


            # Filter data to same amounts per class (against Overfitting)
            df_stable = timestamps_to_use[timestamps_to_use['willFall_dt'] > trigger_range * 10 ** 6]
            df_2 = timestamps_to_use[trigger_range * 10 ** 6 >= timestamps_to_use['willFall_dt']]

            df_stable = df_stable.drop(np.random.choice(df_stable.index, len(df_stable) - len(df_2), replace=False))

            timestamps_to_use = pd.concat([df_stable['ts'], df_2['ts']])

            print_stamped_message(1, f'total: {len(timestamps_to_use)}   '
                f'0: {len(df_stable)}   '
                f'1: {len(df_2)}   ')

            print_stamped_message(1, f'Data set length: {len(timestamps_to_use)}')
            total_data += len(timestamps_to_use)

            #time_steps = 50
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

            inputs['willFallinT'] = inputs.apply(lambda row: 0 if row['willFall_dt'] > trigger_range * (10 ** 6) else 1, axis=1)

            csv_list.append(inputs)
        #break


    print_stamped_message(1,f'total data amount: {total_data}')  # 19875 without Goalie
    df = pd.concat(csv_list, axis=0, ignore_index=True)

    # Constructing the input
    X = df['vals']

    # Constructing the output
    y = df['willFallinT']

    # Split the data into train/vali/test set ~(70/15/15)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.176, random_state=42, shuffle=True)

    # Dump data for later use in model training
    if not os.path.exists(os.path.join(out_dir_path, new_dir)):
        os.makedirs(os.path.join(out_dir_path, new_dir))
    
    out_dir_path = os.path.join(out_dir_path, new_dir)

    with open(os.path.join(out_dir_path, f"ts{time_steps}_range{trigger_range}_xtrain.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    f.close()
    with open(os.path.join(out_dir_path, f"ts{time_steps}_range{trigger_range}_xval.pkl"), "wb") as f:
        pickle.dump(X_val, f)
    f.close()
    with open(os.path.join(out_dir_path, f"ts{time_steps}_range{trigger_range}_xtest.pkl"), "wb") as f:
        pickle.dump(X_test, f)
    f.close()
    with open(os.path.join(out_dir_path, f"ts{time_steps}_range{trigger_range}_ytrain.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    f.close()
    with open(os.path.join(out_dir_path, f"ts{time_steps}_range{trigger_range}_yval.pkl"), "wb") as f:
        pickle.dump(y_val, f)
    f.close()
    with open(os.path.join(out_dir_path, f"ts{time_steps}_range{trigger_range}_ytest.pkl"), "wb") as f:
        pickle.dump(y_test, f)
    f.close()

    print_stamped_message(2, f'saved new model data in {out_dir_path}')

if __name__ == "__main__":
    main()
