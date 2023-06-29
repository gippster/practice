import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('powerconsumption.csv')
print(data.head())
data.info()
data['datetime'] = pd.to_datetime(data['datetime'])
#график изменения электрического потребления со временем
plt.figure(figsize=(12, 6))
plt.plot(data['Datetime'], data['PowerConsumption'])
plt.title('Power Consumption over Time')
plt.xlabel('Datetime')
plt.ylabel('Power Consumption (MWh)')
plt.show()
#распределение температуры
plt.figure(figsize=(8, 6))
sns.histplot(data['Temperature'], bins=20, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()
#распределение значений для "var1"
plt.figure(figsize=(8, 6))
sns.histplot(data['Humidity'], bins=20, kde=True)
plt.title('Humidity Distribution')
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.show()
#гистограмма для "pressure"
plt.figure(figsize=(8, 6))
sns.histplot(data['GeneralDiffuseFlows'], bins=20, kde=True)
plt.title('General Diffuse Flows Distribution')
plt.xlabel('General Diffuse Flows')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['WindSpeed'], bins=20, kde=True)
plt.title('Wind speed Distribution')
plt.xlabel('Wind speed')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['DiffuseFlows'], bins=20, kde=True)
plt.title('Diffuse Flows Distribution')
plt.xlabel('Diffuse Flows')
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
plt.ylabel('Power Consumption')
plt.show()
