import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Завантаження даних 
project_dir = Path(__file__).resolve().parents[2]
csv_path = project_dir / "assets" / "cars.csv"

df = pd.read_csv(csv_path)

print("Набір даних завантажено:", csv_path)
print("\nПерші 5 автомобілів:")
print(df.head())

# Вибір ознак та цільової змінної
X = df[["year", "engine_volume", "mileage", "horsepower"]]

y = df["price"]

# Розділення даних на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nКількість записів для навчання:", len(X_train))
print("Кількість записів для тестування:", len(X_test))

# Створення та навчання моделі
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозування ціни
y_pred = model.predict(X_test)

# Приклад прогнозу для нового автомобіля
new_car = pd.DataFrame([{
    "year": 2020,
    "engine_volume": 2.0,
    "mileage": 80,
    "horsepower": 180
}])

predicted_price = model.predict(new_car)[0]

print("\nПрогноз ціни нового автомобіля:")
print(f"{predicted_price:,.2f} грн")

# Оцінювання якості моделі
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# MAPE — середня абсолютна відсоткова помилка
mape = np.mean(
    np.abs((y_test.to_numpy() - y_pred) / y_test.to_numpy())
) * 100

print("\nРезультати оцінювання моделі:")
print(f"MAE:  {mae:,.2f} грн")
print(f"RMSE: {rmse:,.2f} грн")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Графік: справжня vs прогнозована ціна
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    alpha=0.6,
    label="Прогнози моделі"
)

# Лінія ідеального прогнозу
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    color="red",
    linestyle="--",
    label="Ідеальний прогноз"
)

plt.title("Справжня vs Прогнозована ціна автомобіля")
plt.xlabel("Справжня ціна (грн)")
plt.ylabel("Прогнозована ціна (грн)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
