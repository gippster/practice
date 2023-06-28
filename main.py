import pandas as pd
import numpy as np
import plotly.express as px


df = pd.read_csv('dataset.csv')
print(df)
df.info()
df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
print(df['datetime'])

fig = px.line(df,
              x="Time stamp",
              y="System power generated | (kW)",
              labels = {'Time stamp':'Months of the year'},
              title = "Общая мощность, выработанная с января по декабрь")
fig.show()