import pandas as pd
import matplotlib.pyplot as plt

# Create table with data
data = {
    "OrderId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Customer": [
        "John Smith",
        "Emily Johnson",
        "John Smith",
        "Michael Brown",
        "Emily Johnson",
        "Sarah Wilson",
        "Michael Brown",
        "John Smith",
        "Sarah Wilson",
        "Emily Johnson"
    ],
    "Product": [
        "Laptop",
        "Jacket",
        "Smartphone",
        "Python Book",
        "Monitor",
        "Shoes",
        "AI Book",
        "Tablet",
        "T-Shirt",
        "Data Science Book"
    ],
    "Category": [
        "Electronics",
        "Clothing",
        "Electronics",
        "Books",
        "Electronics",
        "Clothing",
        "Books",
        "Electronics",
        "Clothing",
        "Books"
    ],
    "Quantity": [1, 2, 1, 3, 2, 2, 4, 1, 5, 2],
    "Price": [1200, 150, 800, 200, 300, 120, 180, 600, 80, 250],
    "OrderDate": [
        "2026-06-01",
        "2026-06-03",
        "2026-06-05",
        "2026-06-06",
        "2026-06-07",
        "2026-06-08",
        "2026-06-09",
        "2026-06-10",
        "2026-06-11",
        "2026-06-12"
    ]
}

# конвертація OrderDate
df = pd.DataFrame(data)

df["OrderDate"] = pd.to_datetime(df["OrderDate"])

print("=" * 50 )
print("1. Початкова таблиця з даними")
print("=" * 50)
print(df)

# TotalAmount = Quantity * Price
df["TotalAmount"] = df["Quantity"] * df["Price"]

print("\n" + "=" * 50)
print("2. Таблиця з доданим стовпцем TotalAmount")
print("=" * 50)
print(df)

# Сумарний дохід магазину
total_revenue = df["TotalAmount"].sum()

print("\n" + "=" * 50)
print("3a. Сумарний дохід магазину")
print("=" * 50)
print(f"Total revenue: {total_revenue}")

# Середнє значення TotalAmount
average_total_amount = df["TotalAmount"].mean()

print("\n" + "=" * 50)
print("3b. Середнє значення TotalAmount")
print("=" * 50)
print(f"Average TotalAmount: {average_total_amount:.2f}")

# Кількість замовлень по кожному клієнту
orders_by_customer = df["Customer"].value_counts()

print("\n" + "=" * 50)
print("3c. Кількість замовлень по кожному клієнту")
print("=" * 50)
print(orders_by_customer)

# Замовлення, в яких сума покупки перевищує 500
orders_over_500 = df[df["TotalAmount"] > 500]

print("\n" + "=" * 50)
print("4. Замовлення, в яких сума покупки перевищує 500")
print("=" * 50)
print(orders_over_500)

# Сортування таблиці за OrderDate у зворотному порядку

sorted_orders = df.sort_values(by="OrderDate", ascending=False)

print("\n" + "=" * 50)
print("5. Замовлення, відсортовані за OrderDate у зворотному порядку")
print("=" * 50)
print(sorted_orders)

# Замовлення з 5 по 10 червня включно
start_date = pd.Timestamp("2026-06-05")
end_date = pd.Timestamp("2026-06-10")

orders_from_june_5_to_10 = df[
    (df["OrderDate"] >= start_date) &
    (df["OrderDate"] <= end_date)
]

print("\n" + "=" * 50)
print("6. Замовлення з 5 по 10 червня включно")
print("=" * 50)
print(orders_from_june_5_to_10)



# Групування замовлень за Category
category_summary = df.groupby("Category").agg(
    Quantity=("Quantity", "sum"),
    TotalSales=("TotalAmount", "sum")
)

print("\n" + "=" * 50)
print("7. Групування замовлень за Category")
print("=" * 50)
print(category_summary)



# ТОП-3 клієнтів за загальною сумою покупок
top_3_customers = (df.groupby("Customer")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(3)
)

print("\n" + "=" * 50)
print("8. ТОП-3 клієнтів за загальною сумою покупок")
print("=" * 50)
print(top_3_customers)
