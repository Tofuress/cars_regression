import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv('drom_archive_cleaned_2018-2025.csv')

df["Дата размещения объявления"] = pd.to_datetime(df["Дата размещения объявления"])
df["Год объявления"] = df["Дата размещения объявления"].dt.year
df["Месяц объявления"] = df["Дата размещения объявления"].dt.month
df["Возраст авто"] = df["Год объявления"] - df["Год"]
df = df.drop("Дата размещения объявления", axis=1)
df.info()

categorical = ['Название машины', 'Тип двигателя', 'Коробка передач', 'Привод', 'Руль', 'Поколение', 'Рестайлинг', 'Тип кузова', 'Метка', 'Город']
numerical = ['Год', 'Объем двигателя', 'Мощность', 'Пробег', 'Год объявления', 'Месяц объявления', 'Возраст авто']

preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical), ('num', 'passthrough', numerical)], remainder='drop')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        max_depth=20,
        verbose=2
    ))
])

y = df['Цена']
X = df.drop('Цена', axis=1)

X_train = X[X['Год объявления'] <= 2023]
X_test = X[X['Год объявления'] >= 2024]

y_train = y[X['Год объявления'] <= 2023]
y_test = y[X['Год объявления'] >= 2024]

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rf_mse = mean_squared_error(y_test, y_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

pd.DataFrame({
    'Метод оценки': ['Среднеквадратическая ошибка (MSE)', 'Среднеквадратическая ошибка (RMSE)', 'Средняя абсолютная ошибка (MAE)', 'Коэффицент детерминации (R^2)'],
    'Результаты': [rf_mse, rf_rmse, rf_mae, rf_r2]
})