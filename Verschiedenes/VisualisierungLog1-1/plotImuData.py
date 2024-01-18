import plotly.graph_objects as go

import pandas as pd
#define STATE_INITIAL  0
#define STATE_READY    1
#define STATE_SET      2
#define STATE_PLAYING  3
#define STATE_FINISHED 4

# Load data
df = pd.read_csv("FallPrediction\\DataAcquisition\\DataLabeling\\simpleLabel-isFallen\\data\\1sputnik-11-25-34_labeled.csv")#D:/FallPrediction/fallPredictionCode/outputs/imu_logdata_1_1.csv

# Die erste Zeile als Referenzwert (0) speichern
reference_value = df.iloc[0, 0]
# Spalte 1 entsprechend umwandeln
df['timestamp'] = df['timestamp'] - reference_value
df['timestamp'] = df['timestamp'] / 10**6 #my sec in sekunden umwandeln

df['isFallen'] = df['isFallen'] * 5#für bessere Visualisierung von isFallen

# Create figure
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(df.timestamp), y=list(df.accelX),name="accelX"))
fig.add_trace(go.Scatter(x=list(df.timestamp), y=list(df.accelY),name="accelY"))
fig.add_trace(go.Scatter(x=list(df.timestamp), y=list(df.accelZ),name="accelZ"))
fig.add_trace(go.Scatter(x=list(df.timestamp), y=list(df.gyroYaw),name="gyroYaw"))
fig.add_trace(go.Scatter(x=list(df.timestamp), y=list(df.gyroPitch),name="gyroPitch"))
fig.add_trace(go.Scatter(x=list(df.timestamp), y=list(df.gyroRoll),name="gyroRoll"))
fig.add_trace(go.Scatter(x=list(df.timestamp), y=list(df.isFallen),name="isFallen"))

# Set title
fig.update_layout(
    title_text="Flightlog IMU Data - Player 1 - Log #1" 
)

# Add range slider
fig.update_layout(
    xaxis=dict(
      title='t in s',
      rangeslider=dict(
        visible=True
      ),
      type="linear"
    ),
    yaxis=dict(
      title='Beschleunigung oder Winkelbeschleunigung'
    )
)

fig.show()