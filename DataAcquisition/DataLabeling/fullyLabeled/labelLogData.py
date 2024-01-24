import math
import os
import sys

import pandas as pd


# Funktion zur Berechnung der Zeit bis zum nächsten isFallen = 1 Eintrag
def berechne_dt(dataframe):
    is_fallen_index = dataframe[dataframe['isFallen'] == 1].index
    next_is_fallen_index = is_fallen_index + 1
    next_is_fallen_index = next_is_fallen_index[next_is_fallen_index < len(dataframe)]

    dt_list = []

    for index in is_fallen_index:
        if len(next_is_fallen_index) > 0:
            next_index = next_is_fallen_index[0]
            dt = dataframe.at[next_index, 'timestamp'] - dataframe.at[index, 'timestamp']
            dt_list.append(dt.total_seconds())
            next_is_fallen_index = next_is_fallen_index[1:]
        else:
            # Setze MAX_INT_VALUE, wenn es keinen nächsten isFallen = 1 Eintrag gibt
            dt_list.append(sys.maxsize)

    return dt_list


threshold_rad = 25 * math.pi / 180
src_dir = '.\DataAcquisition\DataLabeling\simpleLabel\data\'

for csv in os.listdir(src_dir):
    output_csv = csv + '-labeled.csv'
    print(output_csv)
    '''
    # Datenrahmen erstellen
    df = pd.read_csv(csv)

    # Neue Spalte 'isFallen' erstellen
    df['isFallen'] = df.apply(lambda row: 1 if abs(row['bodyPitch']) > threshold_rad or abs(row['bodyRoll']) > threshold_rad else 0, axis=1)


    # Neue CSV-Datei speichern, inklusive der 'timestamp'-Spalte
    df.to_csv(csv, index=False, columns=['timestamp', 'gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ', 'isFallen'])
    '''



