import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

data = pd.read_csv('powerconsumption.csv')
print(data.head())
data.info()
data['Datetime'] = pd.to_datetime(data['Datetime'])
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
plt.scatter(data['windspeed'], data['electricity_consumption'])
plt.title('windspeed vs Electricity Consumption')
plt.xlabel('windspeed')
plt.ylabel('Electricity Consumption (MWh)')
plt.show()
