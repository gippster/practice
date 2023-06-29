import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import datetime as dt
import numpy as np
import plotly.figure_factory as ff

data = pd.read_csv('powerconsumption.csv')
print(data.head())
data.info()
data['Datetime'] = pd.to_datetime(data['Datetime'])

#Проверка на пропущенные значения
data.isnull().sum()

#Проверка на дубликаты в данных
data.duplicated().sum()

#Просмотр описательной статистики в наборе данных
data.describe()

#Просмотр описательной статистики в наборе данных с округлением до 2х знаков после запятой
data.describe().round(2)

#Создание нового столбца в наборе данных
data["Month"] = data["Datetime"].dt.month
print(data)

#Установка индекса на столбец Time stamp
data = data.set_index(data["Datetime"])

#Группировка данных по месяцам для более простого и эффективного построения трендов.
grouped = data.groupby('Month').mean(numeric_only=True)
print(grouped)

fig = px.box(data,
        x=data.index.month,
        y="PowerConsumption",
        color=data.index.month, 
        labels = {"x" : "Месяцы"},
        title="Выработка электроэнергии | Месячная статистика ")

fig.update_traces(width=0.5)
fig.show()

fig = px.box(data,
        x=data.index.day,
        y="PowerConsumption",
        color=data.index.day,
        labels = {"x" : "Дни"})

fig.update_traces(width=0.5)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="PowerConsumption",
              labels = {'Month':'Месяцы'},
              color = "PowerConsumption",
              title="Выработка электроэнергии в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="WindSpeed",
              labels = {'Month':'Месяцы'},
              color = "WindSpeed",
              title="Скорость ветра в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="Temperature",
              labels = {'Month':'Month in the year'},
              color = "Temperature",
              title="Температура воздуха в месяц")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.box(data,
             y="PowerConsumption",
             title="Общая статистика выработки электроэнергии")

fig.show()

fig = px.box(data,
             y="WindSpeed",
             title="Общая статистика скорости ветра")

fig.show()

#график изменения электрического потребления со временем

plt.figure(figsize=(12, 6))
plt.plot(data['Datetime'], data['PowerConsumption'])
plt.title('Electrical Consumption over Time')
plt.xlabel('Datetime')
plt.ylabel('Electricity Consumption (MWh)')
plt.show()

#распределение температуры

plt.figure(figsize=(8, 6))
sns.histplot(data['Temperature'], bins=20, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()


#гистограмма для "Humidity"

plt.figure(figsize=(8, 6))
sns.histplot(data['Humidity'], bins=20, kde=True)
plt.title('Humidity Distribution')
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['WindSpeed'], bins=20, kde=True)
plt.title('Windspeed Distribution')
plt.xlabel('WindSpeed')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['GeneralDiffuseFlows'], bins=20, kde=True)
plt.title('GeneralDiffuseFlows Distribution')
plt.xlabel('GeneralDiffuseFlows')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['PowerConsumption'], bins=20, kde=True)
plt.title('Power Consumption Distribution')
plt.xlabel('Power Consumption (MWh)')
plt.ylabel('Count')
plt.show()

# Вычисление корреляционной матрицы
correlation_matrix = data.corr()

# Визуализация корреляций с помощью тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['Temperature'], data['PowerConsumption'])
plt.title('Temperature vs Power Consumption')
plt.xlabel('Temperature')
plt.ylabel('Power Consumption (MWh)')
plt.show()

fig = px.scatter(data,
                 x="Temperature",
                 y="PowerConsumption",
                 title = "Power consumption vs Temperature")
fig.show()
