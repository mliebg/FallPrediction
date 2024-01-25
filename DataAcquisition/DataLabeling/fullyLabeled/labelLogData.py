import math
import os
import sys

import pandas as pd

# Funktion zur Berechnung der Zeit bis zum nächsten isFallen = 1 Eintrag
def calc_dt(dataframe):
    is_fallen_index = dataframe[dataframe['isFallen'] == 1].index

    dt_list = []
    last_index = 0
    for index in is_fallen_index:
        if index - last_index > 1:
            for i in range(last_index, index):
                dt = dataframe.at[index, 'timestamp'] - dataframe.at[i, 'timestamp']
                dt_list.append(dt)
        else:
            dt_list.append(0)
        last_index = index

    return dt_list


threshold_rad = 25 * math.pi / 180
src_dir = r'./DataAcquisition/LogFiltering/filteredLogData/'

for csv in os.listdir(src_dir):
    if csv.endswith('csv'):
        print(csv)

        output_csv = './DataAcquisition/DataLabeling/fullyLabeled/data/' + csv[:len(csv)-4] + '_labeled.csv'

        # Datenrahmen erstellen
        df = pd.read_csv(src_dir + csv)

        # 'isFallen' erstellen
        df['isFallen'] = df.apply(lambda row: 1 if abs(row['bodyPitch']) > threshold_rad or abs(row['bodyRoll']) > threshold_rad else 0, axis=1)

        # 'willFall_dt' erstellen
        dt = calc_dt(df)
        df['willFall_dt'] = dt + [sys.maxsize] * (len(df) - len(dt))
        # Für isFallen = 1 Perioden ist der letzte Eintrag für willFall_dt != 0
        # Q: Ist es sinnvoll für alle isFallen = 1 -> willFall_dt = 0 ODER doch so lassen?

        # Neue CSV-Datei speichern, inklusive isFallen & willFall_dt
        df.to_csv(output_csv, index=False, columns=['timestamp', 'gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ', 'isFallen', 'willFall_dt'])
        ''''''
