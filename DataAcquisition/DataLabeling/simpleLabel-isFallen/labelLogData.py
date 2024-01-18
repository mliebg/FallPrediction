import math
import pandas as pd

# Einlesen der CSV-Datei
input_csv = 'input.csv'
output_csv = 'output.csv'

# Datenrahmen erstellen
df = pd.read_csv(input_csv)

threshold_deg = 25
threshold_rad = threshold_deg * math.pi / 180
# Neue Spalte 'a' erstellen
df['isFallen'] = df.apply(lambda row: 1 if abs(row['bodyPitch']) > threshold_rad or abs(row['bodyRoll']) > threshold_rad else 0, axis=1)

# Neue CSV-Datei speichern, inklusive der 'timestamp'-Spalte
df.to_csv(output_csv, index=False, columns=['timestamp', 'gyroYaw', 'gyroPitch', 'gyroRoll', 'accelX', 'accelY', 'accelZ', 'isFallen'])