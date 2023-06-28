import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('dataset.csv')
print(data.head())
data.info()
data['datetime'] = pd.to_datetime(data['datetime'])
#график изменения электрического потребления со временем
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['electricity_consumption'])
plt.title('Electrical Consumption over Time')
plt.xlabel('Datetime')
plt.ylabel('Electricity Consumption (MWh)')
plt.show()
#распределение температуры
plt.figure(figsize=(8, 6))
sns.histplot(data['temperature'], bins=20, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()
#распределение значений для "var1"
plt.figure(figsize=(8, 6))
sns.histplot(data['var1'], bins=20, kde=True)
plt.title('Var1 Distribution')
plt.xlabel('Var1')
plt.ylabel('Count')
plt.show()
#гистограмма для "pressure"
plt.figure(figsize=(8, 6))
sns.histplot(data['pressure'], bins=20, kde=True)
plt.title('Pressure Distribution')
plt.xlabel('Pressure')
plt.ylabel('Count')
plt.show()

# plt.figure(figsize=(8, 6))
# sns.histplot(data['windspeed'], bins=20, kde=True)
# plt.title('Windspeed Distribution')
# plt.xlabel('Windspeed')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.histplot(data['var2'], bins=20, kde=True)
# plt.title('Var2 Distribution')
# plt.xlabel('Var2')
# plt.ylabel('Count')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.histplot(data['electricity_consumption'], bins=20, kde=True)
# plt.title('Electricity Consumption Distribution')
# plt.xlabel('Electricity Consumption (MWh)')
# plt.ylabel('Count')
# plt.show()

# Вычисление корреляционной матрицы
correlation_matrix = data.drop(columns=['var2','datetime'],axis = 1).corr()

# Визуализация корреляций с помощью тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['temperature'], data['var1'])
plt.title('Temperature vs var1')
plt.xlabel('Temperature')
plt.ylabel('var1')
plt.show()
