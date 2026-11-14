import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from kagglehub import KaggleDatasetAdapter
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Фіксація випадковості
tf.keras.utils.set_random_seed(42)

# Завантаження датасету
def load_data(filename):
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "zalando-research/fashionmnist/versions/4",
        filename,
        pandas_kwargs={"dtype": np.uint8}
    )

    labels = df.pop("label").to_numpy()

    images = df.to_numpy(dtype=np.float32)
    images = images.reshape(-1, 28, 28, 1) / 255.0

    return images, labels

print("Завантаження датасету...")

X_train, y_train = load_data("fashion-mnist_train.csv")
X_test, y_test = load_data("fashion-mnist_test.csv")

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# Перемішування навчальних даних
indices = np.random.default_rng(42).permutation(len(y_train))
X_train = X_train[indices]
y_train = y_train[indices]

# Створення моделі
model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Перший згортковий блок
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Другий згортковий блок
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Перетворення карт ознак у вектор
    layers.Flatten(),

    # Повнозв'язний шар
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Вихідний шар
    layers.Dense(10, activation="softmax")
])

# Компіляція моделі
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Навчання мережі
print("\nПочаток тренування...")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=2
)

# Оцінка моделі
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=2
)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Побудова графіків
epochs = range(1, len(history.history["accuracy"]) + 1)
plt.figure(figsize=(14, 5))

# Графік точності
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["accuracy"], label="Train accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Графік похибки
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history["loss"], label="Train loss")
plt.plot(epochs, history.history["val_loss"], label="Validation loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Матриця помилок
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

predictions = model.predict(X_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(11, 9))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Перевірка прогнозів
random_indices = np.random.default_rng(42).choice(
    len(X_test), size=5, replace=False
)

fig, axes = plt.subplots(1, 5, figsize=(14, 4))

for ax, index in zip(axes, random_indices):
    predicted_class = y_pred[index]
    actual_class = y_test[index]
    ax.imshow(X_test[index].squeeze(), cmap="gray")
    ax.set_title(
        f"Predicted: {class_names[predicted_class]}\n"
        f"Actual: {class_names[actual_class]}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
