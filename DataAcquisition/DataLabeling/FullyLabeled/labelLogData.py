import math
import os
import sys

import pandas as pd


# Funktion zur Berechnung der Zeit bis zum nächsten isFallen = 1 Eintrag
def calc_dt(dataframe):
    """
    Calculates the time in ticks (12ms) until the next time the column isFallen = 1
    :param dataframe:
    :return:
    """
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


threshold_rad = 25 * math.pi / 180  # the threshold that determines if the robot is declared as fallen according to firmware
src_dir = r'./DataAcquisition/LogFiltering/filteredLogData/'

for csv in os.listdir(src_dir):
    if csv.endswith('csv'):
        print(csv)

        df = pd.read_csv(src_dir + csv)

        # create row isFallen. 1 if any of the body angles is greater than 25 degree, 0 else.
        df['isFallen'] = df.apply(lambda row: 1 if abs(row['bodyPitch']) > threshold_rad or abs(row['bodyRoll']) > threshold_rad else 0, axis=1)

        # create willFall_dt column
        dt = calc_dt(df)
        # fill last rows with max integer, bc no more row with isFallen = 1 will appear
        # therefore is: delta t = infinity
        df['willFall_dt'] = dt + [sys.maxsize] * (len(df) - len(dt))

        # Save new CSV with labels isFallen & willFall_dt
        output_csv = './DataAcquisition/DataLabeling/FullyLabeled/data/' + csv[:len(csv)-4] + '_labeled.csv'
        df.to_csv(output_csv, index=False, columns=['timestamp', 'gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ', 'isFallen', 'willFall_dt'])
