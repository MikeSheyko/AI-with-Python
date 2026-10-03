import numpy as np
import matplotlib.pyplot as plt

# Task 1
x = np.linspace(-10, 10, 500)
y = x**2 * np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Графік функції f(x) = x² · sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()


# Task 2
data = np.random.normal(loc=5, scale=2, size=1000)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor="black")
plt.title("Гістограма нормального розподілу")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.grid(axis="y", alpha=0.3)
plt.show()


# Task 3
hobbies = ["Programming", "Gaming", "Music", "Sports", "Reading"]
shares = [33, 27, 11, 19, 10]
plt.figure(figsize=(8, 8))
plt.pie(shares, labels=hobbies, autopct="%1.1f%%", startangle=90)
plt.title("Мої улюблені хобі")
plt.show()


# Task 4
np.random.seed(42)
apples = np.random.normal(loc=180, scale=20, size=100)
bananas = np.random.normal(loc=120, scale=15, size=100)
oranges = np.random.normal(loc=160, scale=18, size=100)
peaches = np.random.normal(loc=150, scale=17, size=100)
fruits = [apples, bananas, oranges, peaches]
plt.figure(figsize=(10, 6))
plt.boxplot(fruits, tick_labels=["Apples", "Bananas", "Oranges", "Peaches"])
plt.title("Маса 100 фруктів кожного виду")
plt.xlabel("Тип фрукту")
plt.ylabel("Маса (г)")
plt.grid(axis="y", alpha=0.3)
plt.show()


# Task 5
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="green", alpha=0.6)
plt.title("Точкова діаграма")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.show()


# Task 6
x = np.linspace(-10, 10, 500)

f = np.sin(x)
g = np.cos(x)
h = np.sin(x) + np.cos(x)

plt.figure(figsize=(10, 6))

plt.plot(x, f, color="blue", label="f(x) = sin(x)")
plt.plot(x, g, color="red", label="g(x) = cos(x)")
plt.plot(x, h, color="green", label="h(x) = sin(x) + cos(x)")

plt.title("Графіки трьох функцій")
plt.xlabel("x")
plt.ylabel("y")

plt.grid(True)
plt.legend()

plt.show()

