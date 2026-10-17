import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Завантажуємо дані
df = pd.read_csv("./assets/fuel_consumption_vs_speed.csv")
print(df.head())

X = df[["speed_kmh"]].values
Y = df["fuel_consumption_l_per_100km"].values


# 2. Перевіряємо різні ступені полінома
best_degree = 3
best_mse = float("inf")
best_model = None

for degree in range(1, 6):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Навчаємо модель
    model.fit(X, Y)

    # Передбачення
    y_pred = model.predict(X)

    # Рахуємо помилки
    mae = mean_absolute_error(Y, y_pred)
    mse = mean_squared_error(Y, y_pred)

    print(f"Ступінь полінома: {degree}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print()

    # Зберігаємо модель з найменшим MSE
    if mse < best_mse:
        best_mse = mse
        best_degree = degree
        best_model = model


# 3. Виводимо найкращу модель
print(f"Найкращий ступінь полінома: {best_degree}")


# 4. Будуємо графік
X_test = np.linspace(X.min(), 140, 500).reshape(-1, 1)
Y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label="Real data", color="blue")
plt.plot(X_test, Y_pred, label=f"Polynomial regression, degree {best_degree}",
         color="red", linestyle="--")

plt.xlabel("Speed (km/h)")
plt.ylabel("Fuel consumption (l/100 km)")
plt.title("Fuel Consumption vs Speed")
plt.legend()
plt.grid(True)
plt.show()


# 5. Прогноз для заданих швидкостей
speeds = [35, 95, 140]

for speed in speeds:
    result = best_model.predict(np.array([[speed]]))
    print(f"Speed = {speed} km/h: {result[0]:.2f} l/100 km")