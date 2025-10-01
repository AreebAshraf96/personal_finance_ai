import os
import pandas as pd
from src.models import (
    train_expense_categorizer,
    predict_category,
    detect_anomalies,
    forecast_expenses,
)

# ---------------- Paths ----------------
BASE = r"C:\Users\hp\personal_finance_ai"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")

cleaned_file = os.path.join(DATA_DIR, "clean_transactions.csv")
model_file = os.path.join(OUT_DIR, "models", "categorizer.pkl")
anomalies_file = os.path.join(OUT_DIR, "anomalies", "anomalies.csv")
forecast_plot = os.path.join(OUT_DIR, "forecast", "forecast_6m.png")
forecast_csv = os.path.join(OUT_DIR, "forecast", "forecast_6m.csv")

# ---------------- Run Tests ----------------
print("🚀 Running Model Tests...\n")

# 1. Train categorizer
print("🔹 Training categorizer...")
train_expense_categorizer(file_path=cleaned_file, model_path=model_file)

# 2. Test prediction
print("🔹 Testing prediction...")
sample_text = "Netflix subscription"
pred = predict_category(sample_text, model_path=model_file)
print(f"Prediction for '{sample_text}': {pred}")

# 3. Run anomaly detection
print("🔹 Running anomaly detection...")
df_anomalies = detect_anomalies(file_path=cleaned_file, save_path=anomalies_file)
print("First 5 rows of anomaly detection results:")
print(df_anomalies.head())

# 4. Forecast expenses (6 months)
print("🔹 Forecasting expenses (next 6 months)...")
forecast_tail = forecast_expenses(
    file_path=cleaned_file,
    periods=6,
    plot_path=forecast_plot,
    csv_path=forecast_csv,
)
print(forecast_tail)

print("\n✅ All model tests completed successfully.")
