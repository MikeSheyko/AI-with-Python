import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Завантаження даних
df = pd.read_csv("./assets/internship_candidates_final_numeric.csv")
print(df.head())

X = df[["Experience", "Grade", "EnglishLevel", "Age", "EntryTestScore"]]
Y = df["Accepted"]

# 2. Розділення даних на навчальні та тестові
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# 3. Створення моделі логістичної регресії
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 4. Навчання моделі
model.fit(X_train, Y_train)

# 5. Перевірка точності моделі
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 6. Передбачення ймовірності прийняття
student = pd.DataFrame([{
    "Experience": 3,
    "Grade": 10.5,
    "EnglishLevel": 3,
    "Age": 22,
    "EntryTestScore": 850
}])

probability = model.predict_proba(student)[0][1]
prediction = model.predict(student)[0]

print(f"Probability of acceptance: {probability:.2%}")

if prediction == 1:
    print("Student will be accepted for internship.")
else:
    print("Student will not be accepted for internship.")

# 7. Візуалізація результатів моделі
plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["Grade"],
    X_test["EntryTestScore"],
    c=Y_pred,
    cmap="coolwarm",
    edgecolor="k",
    s=100
)

plt.title("Logistic Regression Predictions")
plt.xlabel("GPA")
plt.ylabel("Entry Test Score")
plt.colorbar(label="Predicted Class")
plt.grid(True)
plt.show()