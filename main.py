import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import datetime as dt
import numpy as np
import plotly.figure_factory as ff

data = pd.read_csv('powerconsumption_original.csv')
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
             y="Temperature",
             title="Общая статистика температуры")

fig.show()

# график изменения электрического потребления со временем

plt.figure(figsize=(12, 6))
plt.plot(data['Datetime'], data['PowerConsumption'])
plt.title('Electrical Consumption over Time')
plt.xlabel('Datetime')
plt.ylabel('Electricity Consumption (KWh)')
plt.show()

# распределение температуры

plt.figure(figsize=(8, 6))
sns.histplot(data['Temperature'], bins=20, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()


# гистограмма для "Humidity"

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
plt.xlabel('Power Consumption (KWh)')
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
plt.ylabel('Power Consumption (KWh)')
plt.show()

fig = px.scatter(data,
                 x="Temperature",
                 y="PowerConsumption",
                 title = "Power consumption vs Temperature")
fig.show()

data_1_to_11 = data[data['Month'].isin(range(1, 12))] # Select rows with months 1 to 11
data_12 = data[data['Month'] == 12] # Select rows with month 12

# Print the first 5 rows of each dataframe
data_1_to_11.to_csv("powerconsumption.csv", index=False)
data_12.to_csv("pr.csv", index=False)
print(data_1_to_11.head())
print(data_12.head())



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostRegressor 


data_1_to_11 = data_1_to_11.drop('Datetime', axis = 1)

# Разделение датасета на обучение и тесты
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Define the features and target variable
features = ["Temperature",  "Humidity",  "WindSpeed",  "DiffuseFlows", "PowerConsumption",  "Month"]
target = "PowerConsumption"

# Обучение XGBoost модели
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.02,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.3,
    n_estimators=1000,
    random_state=42
)

xgb_model.fit(train_df[features], train_df[target])
xgb_preds = xgb_model.predict(test_df[features])

# Обучение CatBoost модели
cat_model = CatBoostRegressor(
    learning_rate=0.01,
    depth=10,
    iterations=1000,
    verbose=False,
    random_state=42
)

cat_model.fit(train_df[features], train_df[target])
cat_preds = cat_model.predict(test_df[features])


data_12 = data_12.drop('Datetime', axis = 1)

next_24_hours_X = data_12[["Temperature",  "Humidity",  "WindSpeed",  "DiffuseFlows", "PowerConsumption",  "Month"]]

# Make predictions using the model
next_24_hours_preds_xgb = xgb_model.predict(next_24_hours_X)
next_24_hours_preds_cat = cat_model.predict(next_24_hours_X)

# Add the predicted values to the original dataframe
data_12["PowerConsumptionXGB"] = next_24_hours_preds_xgb.flatten()
data_12["PowerConsumptionCat"] = next_24_hours_preds_cat.flatten()

# Write the updated data to the same excel file
data_12.to_csv("PredictData.csv", index=False)

data = pd.read_csv("pr.csv")
datapredict = pd.read_csv("PredictData.csv")
mapedf = np.mean(np.abs((data["PowerConsumption"] - datapredict["PowerConsumptionXGB"]) / data["PowerConsumption"])) * 100
mape = np.mean(np.abs((test_df[target] - xgb_preds) / test_df[target])) * 100
mae = mean_absolute_error(test_df[target], xgb_preds)
mse = mean_squared_error(test_df[target], xgb_preds)
rmse = np.sqrt(mse)
r2 = r2_score(test_df[target], xgb_preds)
# Вывод XGBoost
print("Метрики работы модели XGBoost:")
print("__________________________________________________________________")
print("Model Percentage Mean Absolute Error: ", mape)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R^2: ", r2)
print("Percentage Mean Absolute Error: ", mapedf)
print("__________________________________________________________________")


mapedf = np.mean(np.abs((data["PowerConsumption"] - datapredict["PowerConsumptionCat"]) / data["PowerConsumption"])) * 100
mape = np.mean(np.abs((test_df[target] - cat_preds) / test_df[target])) * 100
mae = mean_absolute_error(test_df[target], cat_preds)
mse = mean_squared_error(test_df[target], cat_preds)
rmse = np.sqrt(mse)
r2 = r2_score(test_df[target], cat_preds)
# Вывод CatBoost
print("Метрики работы модели Catboost:")
print("__________________________________________________________________")
print("Model Percentage Mean Absolute Error: ", mape)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R^2: ", r2)
print("Percentage Mean Absolute Error: ", mapedf)
print("__________________________________________________________________")