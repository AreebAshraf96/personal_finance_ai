# src/models.py
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
BASE = r"C:\Users\hp\personal_finance_ai"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
MODELS_DIR = os.path.join(OUT_DIR, "models")
ANOMALIES_DIR = os.path.join(OUT_DIR, "anomalies")
FORECAST_DIR = os.path.join(OUT_DIR, "forecast")

for d in [MODELS_DIR, ANOMALIES_DIR, FORECAST_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------- Expense Categorizer ----------------
def train_expense_categorizer(
    file_path=os.path.join(DATA_DIR, "clean_transactions.csv"),
    model_path=os.path.join(MODELS_DIR, "categorizer.pkl"),
):
    """
    Train a text-based classifier to predict expense categories from Description.
    Saves (model, vectorizer) with joblib at model_path.
    """
    df = pd.read_csv(file_path)

    # keep only expense rows
    if "Type" in df.columns:
        df = df[df["Type"] == "Expense"].copy()

    if df.empty or "Description" not in df.columns or "Category" not in df.columns:
        print("⚠️ Cannot train categorizer (no expense rows or missing columns).")
        return None, None

    X = df["Description"].astype(str)
    y = df["Category"].astype(str)

    vec = TfidfVectorizer()
    X_vec = vec.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y)

    joblib.dump((clf, vec), model_path)
    print(f"✅ Expense categorizer trained and saved to {model_path}")
    return clf, vec


def predict_category(
    description: str,
    model_path=os.path.join(MODELS_DIR, "categorizer.pkl"),
) -> str:
    """Predict category for a new expense description using the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model not found. Train the categorizer first.")
    clf, vec = joblib.load(model_path)
    X_new = vec.transform([description])
    return clf.predict(X_new)[0]


# ---------------- Anomaly Detection ----------------
def detect_anomalies(
    file_path=os.path.join(DATA_DIR, "clean_transactions.csv"),
    save_path=os.path.join(ANOMALIES_DIR, "anomalies.csv"),
):
    """
    Detect unusual spending amounts using Isolation Forest and save results to CSV.
    Adds 'Anomaly' column with values {'Normal','Anomaly'}.
    """
    df = pd.read_csv(file_path)

    if "Amount" not in df.columns or df.empty:
        print("⚠️ No data/Amount column for anomaly detection.")
        return df

    # ensure numeric
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    iso = IsolationForest(contamination=0.05, random_state=42)

    df["Anomaly"] = pd.Series(
        iso.fit_predict(df[["Amount"]]), index=df.index
    ).map({1: "Normal", -1: "Anomaly"})

    df.to_csv(save_path, index=False)
    print(f"✅ Anomaly detection complete. Results saved to {save_path}")
    return df


# ---------------- Forecasting ----------------
def forecast_expenses(
    file_path=os.path.join(DATA_DIR, "clean_transactions.csv"),
    periods: int = 3,
    interval_width: float = 0.95,
    plot_path: str | None = None,
    csv_path: str | None = None,
):
    """
    Forecast future monthly expenses using Prophet.
    Always saves plot (.png) and forecast data (.csv).
    Returns the last `periods` rows of forecast DataFrame.
    """
    if plot_path is None:
        plot_path = os.path.join(FORECAST_DIR, f"forecast_{periods}m.png")
    if csv_path is None:
        csv_path = os.path.join(FORECAST_DIR, f"forecast_{periods}m.csv")

    df = pd.read_csv(file_path)

    if "Date" not in df.columns or "Amount" not in df.columns:
        raise ValueError("Dataset must contain 'Date' and 'Amount' columns.")

    # ensure datetime and numeric
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # only expenses
    df = df[df["Amount"] < 0].dropna(subset=["Date"]).copy()
    if df.empty:
        raise ValueError("No expense rows available for forecasting.")

    # monthly totals
    monthly = (
        df.groupby(df["Date"].dt.to_period("M"))["Amount"]
        .sum()
        .abs()
        .reset_index()
        .rename(columns={"Date": "ds", "Amount": "y"})
    )
    monthly["ds"] = monthly["ds"].dt.to_timestamp()

    if len(monthly) < 3:
        raise ValueError("Not enough monthly points to fit a forecast (need ≥ 3).")

    m = Prophet(interval_width=interval_width)
    m.fit(monthly)

    future = m.make_future_dataframe(periods=periods, freq="MS")
    fc = m.predict(future)

    # save plot
    fig = m.plot(fc)
    plt.title(f"Monthly Expense Forecast ({periods} months, CI={int(interval_width*100)}%)")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # save csv
    tail = fc.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tail.to_csv(csv_path, index=False)

    print(f"✅ Forecast saved: plot={plot_path}, csv={csv_path}")
    return tail


# ---------------- Runnable ----------------
if __name__ == "__main__":
    print("🔹 Training expense categorizer…")
    model, vec = train_expense_categorizer()
    if model is not None:
        sample = "Uber ride to office"
        print(f"Prediction for '{sample}': {predict_category(sample)}")

    print("\n🔹 Running anomaly detection…")
    anom_df = detect_anomalies()
    print(anom_df.head())

    print("\n🔹 Running expense forecast (next 3 months)…")
    tail = forecast_expenses(periods=3, interval_width=0.95)
    print(tail)


