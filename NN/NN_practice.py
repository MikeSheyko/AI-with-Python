import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Створення даних
np.random.seed(42)
tf.random.set_seed(42)

X = np.random.rand(1000, 2).astype("float32")

y = (X[:, 0] + X[:, 1] >= 1).astype("float32")

print("Кількість точок:", len(X))
print("Клас 0:", np.sum(y == 0))
print("Клас 1:", np.sum(y == 1))

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nНавчальна вибірка:", X_train.shape)
print("Тестова вибірка:", X_test.shape)

# Створення нейронної мережі
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Налаштування моделі
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Навчання моделі
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
print("\nНавчання завершено!")

# Перевірка точності
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nРезультати тестування:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Передбачення класів
predictions = model.predict(X_test, verbose=0)
predicted_classes = (predictions >= 0.5).astype("int32").flatten()

print("\nПередбачення для тестових точок:")
print("-" * 65)
print(f"{'X1':<12}{'X2':<12}{'Actual':<12}{'Predicted':<12}")
print("-" * 65)

for i in range(len(X_test)):
    print(
        f"{X_test[i, 0]:<12.4f}"
        f"{X_test[i, 1]:<12.4f}"
        f"{int(y_test[i]):<12}"
        f"{predicted_classes[i]:<12}"
    )

print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

# Перевірка на нових точках
new_points = np.array([
    [0.2, 0.3],
    [0.8, 0.5],
    [0.1, 0.9],
    [0.7, 0.7],
    [0.4, 0.4]
], dtype="float32")

new_predictions = model.predict(new_points, verbose=0)
new_classes = (new_predictions >= 0.5).astype("int32").flatten()

print("\nРезультати для нових точок:")
print("-" * 65)
print(f"{'X1':<12}{'X2':<12}{'Probability':<16}{'Class'}")
print("-" * 65)

for i in range(len(new_points)):
    print(
        f"{new_points[i, 0]:<12.2f}"
        f"{new_points[i, 1]:<12.2f}"
        f"{new_predictions[i][0]:<16.4f}"
        f"{new_classes[i]}"
    )

#  Loss
plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Accuracy
plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Accuracy During Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Графік класифікації точок
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=predicted_classes,
    cmap="coolwarm",
    alpha=0.7,
    label="Predicted Classes"
)

plt.plot([0, 1], [1, 0], "k--", label="Decision Boundary")

plt.title("Neural Network Classification")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()