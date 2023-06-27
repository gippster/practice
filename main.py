import pandas as pd

df = pd.read_csv('dataset.csv')
print(df)
df.info()
df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
print(df['datetime'])
df.info()