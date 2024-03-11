from random import random

import pandas as pd
import numpy as np
import os

# Importing the data
dir_path = './DataAcquisition/DataLabeling/FullyLabeled/data'
out_dir = './DataAcquisition/DataLabeling/sequenceInputs/data/'
csv_list = []


def create_input_sequences(data, time_steps, timestamps):
    """
    Erstellt Input-Sequenzen für ein KNN aus den gegebenen Daten.

    Args:
    - data: DataFrame mit den Daten, timestamp als Index
    - time_steps: Anzahl der vergangenen Zeitpunkte, die berücksichtigt werden sollen

    Returns:
    - sequences: DataFrame mit Input-Sequenzen
    """
    sequences = []
    print(f'data count: {len(data)}')
    progress = 0
    for ts in timestamps:
        print(f'progress: {progress} / {len(timestamps)} ', end='\r')
        i = data.index[data['timestamp'] == ts][0]
        if i < 200 or i > len(data):
            progress += 1
            continue  # Nicht vollständige Einträge überspringen

        sequence = []
        for j in range(time_steps):
            sequence += list(data.iloc[i - j][['gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ']])
        sequence += list(data.iloc[i][['willFall']])#'willFall_dt',
        sequences.append(sequence)

        progress += 1

    # Spaltennamen
    cols = []
    for i in range(time_steps):
        cols.append(f'gyroYaw-{i}')
        cols.append(f'gyroPitch-{i}')
        cols.append(f'gyroRoll-{i}')
        cols.append(f'accelX-{i}')
        cols.append(f'accelY-{i}')
        cols.append(f'accelZ-{i}')

    #cols.append('willFall_dt')
    cols.append('willFall')

    return pd.DataFrame(sequences, columns=cols)


time_steps = 200  # Wie viel Einträge zurückgeschaut werden sollen
for csv in os.listdir(dir_path):
    if csv.endswith(".csv"):
        print(f'next csv: {csv}')
        csv_path = os.path.join(dir_path, csv)
        dataframe = pd.read_csv(csv_path)

        # willFall_dt = 0 kann entfernt werden, da der Roboter während des Sturzes/Aufstehens nicht prüfen muss, ob er fällt
        dataframe = dataframe[dataframe['willFall_dt'] != 0]

        # Klassifikation Outputs
        # 0 : stabil
        # 1 : in 2s
        # 2 : in 1s
        # 3 : in 0.5s
        # 4 : in 0.2s
        dataframe['willFall'] = dataframe.apply(lambda row:
                                                0 if row['willFall_dt'] > 2 * (10 ** 6)
                                                else 1 if row['willFall_dt'] > 1 * (10 ** 6)
                                                else 2 if row['willFall_dt'] > .5 * (10 ** 6)
                                                else 3 if row['willFall_dt'] > .2 * (10 ** 6)
                                                else 4, axis=1)

        # 50/50 für willFall_2s und "wird nicht fallen", gegen Overfitting + spart Rechenaufwand für Umwandlung
        pos_sets = dataframe[dataframe['willFall'] != 0]
        neg_sets = dataframe[dataframe['willFall'] == 0]
        to_remove = len(neg_sets) - len(pos_sets)
        neg_sets = neg_sets.drop(np.random.choice(neg_sets.index, to_remove, replace=False))
        ts_to_takeover = pd.concat([pos_sets['timestamp'], neg_sets['timestamp']])

        # Inputs in time series umwandeln
        input_sequences = create_input_sequences(dataframe, time_steps, ts_to_takeover)

        # Neue CSV speichern
        out_file = csv[4:-12] + '_seq.csv'
        input_sequences.to_csv(out_dir + out_file, index=False)
        print(out_file)
        #input()
