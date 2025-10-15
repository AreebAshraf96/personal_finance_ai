# src/dashboard.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot_expenses_by_category, plot_expenses_over_time
from models import detect_anomalies, predict_category, forecast_expenses

import streamlit as st
import toml
import os
import shutil
# ---------------- Theme Switcher ----------------
def apply_theme(theme_choice: str):
    """
    Dynamically apply Streamlit theme by swapping .streamlit/config.toml
    between light and dark versions.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    streamlit_dir = os.path.join(base_dir, ".streamlit")
    config_path = os.path.join(streamlit_dir, "config.toml")
    theme_file = os.path.join(streamlit_dir, f"config_{theme_choice.lower()}.toml")

    if os.path.exists(theme_file):
        shutil.copy(theme_file, config_path)
    else:
        st.warning(f"⚠️ Theme file '{theme_file}' not found. Using default theme.")

# Sidebar theme switcher
theme_choice = st.sidebar.radio("🎨 Select Theme", ["Dark", "Light"], index=0)
apply_theme(theme_choice)

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Personal Finance AI", layout="wide")
st.title("💰 Personal Finance AI Dashboard")

# ---------------- Paths ----------------
BASE = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_FILE = os.path.join(BASE, "data", "clean_transactions.csv")

PLOTS_DIR = os.path.join(BASE, "outputs", "plots")
ANOMALIES_FILE = os.path.join(BASE, "outputs", "anomalies", "anomalies.csv")
FORECAST_DIR = os.path.join(BASE, "outputs", "forecast")
TMP_DIR = os.path.join(BASE, "outputs", "tmp")

for d in [PLOTS_DIR, FORECAST_DIR, TMP_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------- Sidebar: About / Instructions ----------------
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown(
        """
        **Personal Finance AI** helps you track and understand your financial activity.

        **How to use:**
        1. 📂 Upload a CSV file of your transactions  
           - Must include at least a **Date** and **Amount** column.  
           - Optional: **Description**, **Category**, or **Type** columns.
        2. 🧹 The app automatically detects column names and formats.
        3. 🕵️‍♀️ Explore your data with filters, charts, and anomaly detection.
        4. 📈 Forecast future expenses or income using AI.
        5. 🧾 Generate a professional PDF summary report.

        **Tips:**
        - Click **Reset Filters** to start fresh.
        - Use **Download Filtered Data** to export your current view.
        - Supported encodings: UTF-8 and Latin-1.
        """
    )
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit and Python.")

# ---------------- Helper: flexible column detector ----------------
def detect_column(df, keywords):
    """Return first column name containing any keyword."""
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            return col
    return None
# ---------------- Data Validation Helper ----------------
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate uploaded dataset safely before analysis."""

    # Detect basic columns
    date_col = detect_column(df, ["date", "time", "day"])
    amount_col = detect_column(df, ["amount", "value", "money", "transaction", "debit", "credit"])
    category_col = detect_column(df, ["category", "type"])
    desc_col = detect_column(df, ["desc", "merchant", "details"])

    # ---------- Date ----------
    if date_col:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df["Date"] = df["Date"].dt.tz_localize(None)
    else:
        st.error("❌ No column containing 'date', 'time' or 'day' found.")
        st.stop()

    # ---------- Amount ----------
    if amount_col:
        df["Amount"] = pd.to_numeric(df[amount_col], errors="coerce")
    elif {"debit", "credit"}.issubset({c.lower() for c in df.columns}):
        df["Amount"] = df.get("Credit", 0).fillna(0) - df.get("Debit", 0).fillna(0)
    else:
        st.error("❌ No amount/debit/credit column found in your file.")
        st.stop()

    # ---------- Category ----------
    if category_col:
        df["Category"] = df[category_col]
    else:
        df["Category"] = "Uncategorized"

    # ---------- Description ----------
    if desc_col:
        df["Description"] = df[desc_col]
    else:
        df["Description"] = "Unknown"

    # ---------- Type ----------
    if "Type" not in df.columns:
        df["Type"] = df["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")

    # ---------- Clean invalid rows ----------
    df = df.dropna(subset=["Date", "Amount"]).copy()
    df["Amount"] = df["Amount"].clip(-1e6, 1e6)  # prevent outliers from breaking graphs

    # ---------- Safety summaries ----------
    st.write("✅ Detected Columns:", list(df.columns))
    st.write(f"📊 Rows after cleaning: {len(df)}")

    if df["Amount"].isna().all():
        st.warning("⚠️ All Amount values are NaN — check your numeric formatting.")
    if df["Date"].isna().all():
        st.warning("⚠️ No valid Date values found — check date format (e.g., DD/MM/YYYY).")

    return df

# ---------------- Load default data ----------------
@st.cache_data
def load_default_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)

    # Detect and clean date
    date_col = detect_column(df_, ["date", "time", "day"])
    if date_col:
        df_["Date"] = pd.to_datetime(df_[date_col], errors="coerce").dt.tz_localize(None)
    else:
        st.error("❌ No date-like column found in dataset.")
        st.stop()

    # Detect and clean amount
    amount_col = detect_column(df_, ["amount", "value", "money", "transaction", "debit", "credit"])
    if amount_col:
        df_["Amount"] = pd.to_numeric(df_[amount_col], errors="coerce")
    elif set(["debit", "credit"]).issubset({c.lower() for c in df_.columns}):
        # If both debit/credit exist, subtract
        df_["Amount"] = df_.get("Credit", 0).fillna(0) - df_.get("Debit", 0).fillna(0)
    else:
        st.error("❌ No amount/debit/credit column found in dataset.")
        st.stop()

    # Create Type if missing
    if "Type" not in df_.columns:
        df_["Type"] = df_["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")
    df_ = validate_dataset(df_)

    return df_

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

    # Validate and clean the dataset after loading
    df = validate_dataset(df)



    # Flexible Date detection
    date_col = detect_column(df, ["date", "time", "day"])
    if date_col:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    else:
        st.error("❌ No date-like column found in uploaded file.")
        st.stop()

    # Flexible Amount detection
    amount_col = detect_column(df, ["amount", "value", "money", "transaction", "debit", "credit"])
    if amount_col:
        df["Amount"] = pd.to_numeric(df[amount_col], errors="coerce")
    elif set(["debit", "credit"]).issubset({c.lower() for c in df.columns}):
        df["Amount"] = df.get("Credit", 0).fillna(0) - df.get("Debit", 0).fillna(0)
    else:
        st.error("❌ No amount/debit/credit column found in uploaded file.")
        st.stop()

    # Add Type column
    if "Type" not in df.columns:
        df["Type"] = df["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")

    st.success("✅ File uploaded successfully!")

else:
    st.info("Using default dataset...")
    df = load_default_data(DATA_FILE)

# ---------------- Filters ----------------
st.subheader("🔎 Filters")

# Ensure date column is timezone-naive
if pd.api.types.is_datetime64tz_dtype(df["Date"]):
    df["Date"] = df["Date"].dt.tz_localize(None)

min_d, max_d = df["Date"].min(), df["Date"].max()
col_f1, col_f2 = st.columns(2)
date_range = col_f1.date_input(
    "Date range",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date(),
)
categories = sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else []
selected_cats = col_f2.multiselect("Categories", options=categories, default=categories)

if st.button("🔄 Reset Filters"):
    st.experimental_rerun()


# Apply filters safely
df_filt = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if pd.api.types.is_datetime64tz_dtype(df_filt["Date"]):
        df_filt["Date"] = df_filt["Date"].dt.tz_localize(None)
    df_filt = df_filt[(df_filt["Date"] >= start_d) & (df_filt["Date"] <= end_d)]

if selected_cats:
    df_filt = df_filt[df_filt["Category"].isin(selected_cats)]


# Download filtered data
st.download_button(
    "📥 Download Filtered Data (CSV)",
    data=df_filt.to_csv(index=False).encode("utf-8"),
    file_name="filtered_transactions.csv",
    mime="text/csv",
)


# ---------------- KPIs ----------------
st.subheader("📊 Financial Summary")

total_income = df_filt[df_filt["Amount"] > 0]["Amount"].sum()
total_expenses = df_filt[df_filt["Amount"] < 0]["Amount"].sum() * -1
net_savings = total_income - total_expenses
savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("💵 Total Income", f"${total_income:,.2f}")
col2.metric("💸 Total Expenses", f"${total_expenses:,.2f}")
col3.metric("💰 Net Savings", f"${net_savings:,.2f}")
col4.metric("📈 Savings Rate", f"{savings_rate:.1f}%")

# ---------------- Monthly Breakdown: Income vs Expenses ----------------
st.subheader("📅 Monthly Breakdown: Income vs Expenses")

if not df_filt.empty:
    monthly = df_filt.groupby([df_filt["Date"].dt.to_period("M"), "Type"])["Amount"].sum().reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    pivot_ie = monthly.pivot(index="Date", columns="Type", values="Amount").fillna(0)
    pivot_ie["Income"] = pivot_ie.get("Income", 0)
# Ensure we always have a pandas Series before calling .abs()
    pivot_ie["Expense"] = (
       pivot_ie["Expense"].abs()
       if "Expense" in pivot_ie.columns
       else pd.Series(0, index=pivot_ie.index)
)


    fig1, ax1 = plt.subplots(figsize=(8, 4))
    pivot_ie[["Income", "Expense"]].plot(ax=ax1, marker="o")
    ax1.set_title("Monthly Income vs Expenses")
    ax1.set_ylabel("Amount ($)")
    ax1.set_xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
else:
    st.info("No data available for monthly breakdown with current filters.")

# ---------------- Savings Rate Trend ----------------
st.subheader("📉 Savings Rate Trend (%)")
if not df_filt.empty:
    monthly_net = df_filt.groupby(df_filt["Date"].dt.to_period("M"))["Amount"].sum().to_timestamp()
    monthly_inc = df_filt[df_filt["Amount"] > 0].groupby(df_filt["Date"].dt.to_period("M"))["Amount"].sum().to_timestamp()
    sr = ((monthly_net.clip(lower=0) / monthly_inc) * 100).replace([pd.NA, pd.NaT], 0).fillna(0)
    sr = sr.reindex(monthly_inc.index, fill_value=0)

    fig_sr, ax_sr = plt.subplots(figsize=(8, 3))
    sr.plot(ax=ax_sr, marker="o")
    ax_sr.set_title("Savings Rate by Month")
    ax_sr.set_ylabel("Savings Rate (%)")
    ax_sr.set_xlabel("Month")
    plt.xticks(rotation=45)
    plt.ylim(0, max(100, sr.max() if sr.size else 100))
    plt.tight_layout()
    st.pyplot(fig_sr)
else:
    st.info("No data available for savings rate with current filters.")

# ---------------- Data Preview ----------------
st.subheader("📄 Transactions Preview")
df_preview = df_filt.head().copy()
if "Amount" in df_preview.columns:
    df_preview["Amount"] = df_preview["Amount"].apply(lambda x: f"${x:,.2f}")
st.dataframe(df_preview)

# ---------------- Category Insights ----------------
st.subheader("🧠 Category Insights")

# Bar & line charts (saved to images) using filtered data
cat_plot = os.path.join(PLOTS_DIR, "cat_plot.png")
time_plot = os.path.join(PLOTS_DIR, "time_plot.png")
plot_expenses_by_category(df_filt, cat_plot)
plot_expenses_over_time(df_filt, time_plot)

left_col, right_col = st.columns(2)
with left_col:
    st.image(cat_plot, caption="Expenses by Category (Bar)")
with right_col:
    st.image(time_plot, caption="Expenses Over Time (Line)")

# ---------------- Pie chart of category distribution (expenses only) ----------------
st.subheader("📊 Expense Distribution")
if "Category" not in df_filt.columns:
    cat_col = next((c for c in df_filt.columns if "category" in c.lower() or "type" in c.lower()), None)
    if cat_col:
        df_filt["Category"] = df_filt[cat_col]
    else:
        st.warning("No category-like column found. Pie chart unavailable.")
        df_filt["Category"] = "Uncategorized"

df_exp_only = df_filt[df_filt["Amount"] < 0].copy()

if not df_exp_only.empty:
    cat_totals = (
        df_exp_only.groupby("Category")["Amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    ax_pie.pie(cat_totals.values, labels=cat_totals.index, autopct="%1.1f%%", startangle=140)
    ax_pie.axis("equal")
    ax_pie.set_title("Expense Distribution by Category")
    st.pyplot(fig_pie)
else:
    st.info("No expense data available for pie chart.")

# ---------------- Top 5 categories / merchants by spend ----------------
st.subheader("🏆 Top Spenders")
col_t1, col_t2 = st.columns(2)

if not df_exp_only.empty:
    # --- Top categories ---
    top_cats = (
        df_exp_only.groupby("Category")["Amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    top_cats["Total Spent"] = top_cats["Amount"].apply(lambda x: f"${x:,.2f}")
    top_cats = top_cats[["Category", "Total Spent"]]

    # --- Top merchants (or descriptions) ---
    desc_col = next((c for c in df_exp_only.columns if "desc" in c.lower() or "merchant" in c.lower()), None)
    if desc_col:
        top_merchants = (
            df_exp_only.groupby(desc_col)["Amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        top_merchants.columns = ["Description", "Amount"]
        top_merchants["Total Spent"] = top_merchants["Amount"].apply(lambda x: f"${x:,.2f}")
        top_merchants = top_merchants[["Description", "Total Spent"]]
    else:
        top_merchants = pd.DataFrame(columns=["Description", "Total Spent"])

    # --- Display side-by-side ---
    with col_t1:
        st.write("**Top 5 Categories**")
        st.dataframe(top_cats)
        st.download_button(
            "📥 Download Top Categories (CSV)",
            data=top_cats.to_csv(index=False).encode("utf-8"),
            file_name="top_categories.csv",
            mime="text/csv",
        )

    with col_t2:
        st.write("**Top 5 Merchants / Descriptions**")
        st.dataframe(top_merchants)
        st.download_button(
            "📥 Download Top Merchants (CSV)",
            data=top_merchants.to_csv(index=False).encode("utf-8"),
            file_name="top_merchants.csv",
            mime="text/csv",
        )

else:
    st.info("No expense data available for Top Spenders.")


# ---------------- Anomaly Detection (on filtered data) ----------------
st.subheader("⚠️ Anomaly Detection")

# Save filtered data temporarily, run anomalies on that file
tmp_filtered = os.path.join(TMP_DIR, "filtered_for_anom.csv")
df_filt.to_csv(tmp_filtered, index=False)

df_anom = detect_anomalies(file_path=tmp_filtered, save_path=ANOMALIES_FILE)
anom_rows = df_anom[df_anom["Anomaly"] == "Anomaly"].copy()

if not anom_rows.empty:
    st.warning(f"⚠️ Found {len(anom_rows)} anomalies in filtered data.")
    if "Amount" in anom_rows.columns:
        anom_rows["Amount"] = anom_rows["Amount"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(anom_rows.head(20))

    with open(ANOMALIES_FILE, "rb") as f:
        st.download_button(
            label="📥 Download Anomaly Report (Filtered)",
            data=f,
            file_name="anomalies_filtered.csv",
            mime="text/csv",
        )
else:
    st.success("✅ No anomalies detected for current filters.")

# ---------------- Category Prediction ----------------
st.subheader("🔮 Category Prediction")
desc = st.text_input("Enter an expense description (e.g., Uber ride, Netflix, KFC)")
if desc:
    try:
        pred = predict_category(desc)
        st.success(f"Predicted category: **{pred}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- Forecasting ----------------
st.subheader("📈 Expense Forecast")

horizon = st.selectbox(
    "Select forecast period:",
    options=[3, 6, 12],
    index=0,
    format_func=lambda x: f"{x} months",
)

ci_choice = st.selectbox("Confidence interval:", options=["95%", "90%", "80%"], index=0)
ci_map = {"95%": 0.95, "90%": 0.90, "80%": 0.80}
ci = ci_map[ci_choice]

# Save filtered data temporarily for forecasting on current slice
tmp_forecast = os.path.join(TMP_DIR, "filtered_for_forecast.csv")
df_filt.to_csv(tmp_forecast, index=False)

try:
    forecast = forecast_expenses(
        file_path=tmp_forecast,
        periods=horizon,
        interval_width=ci,
        plot_path=os.path.join(FORECAST_DIR, f"forecast_{horizon}m_{int(ci*100)}ci.png"),
        csv_path=os.path.join(FORECAST_DIR, f"forecast_{horizon}m_{int(ci*100)}ci.csv"),
    )

    st.image(
        os.path.join(FORECAST_DIR, f"forecast_{horizon}m_{int(ci*100)}ci.png"),
        caption=f"Forecast for next {horizon} months (CI {ci_choice})"
    )

    # Pretty forecast table
    forecast_fmt = forecast.copy()
    forecast_fmt["ds"] = pd.to_datetime(forecast_fmt["ds"]).dt.strftime("%b %Y")
    forecast_fmt["Prediction"] = forecast_fmt["yhat"].apply(lambda x: f"${x:,.2f}")
    forecast_fmt["Range"] = forecast_fmt.apply(
        lambda row: f"${row['yhat_lower']:,.2f} – ${row['yhat_upper']:,.2f}", axis=1
    )
    forecast_fmt = forecast_fmt[["ds", "Prediction", "Range"]]

    st.write(f"📌 Predicted expenses for the next {horizon} months:")
    st.dataframe(forecast_fmt)

    # Annual projection (if horizon == 12)
    if horizon == 12:
        annual_total = float(forecast["yhat"].sum())
        st.metric("📅 Projected Yearly Spend", f"${annual_total:,.2f}")

    # What-if slider: adjust forecast by %
    st.subheader("🧪 What-if: Adjust forecast")
    adj = st.slider("Adjust expenses by (%)", min_value=-30, max_value=30, value=0, step=1)
    if adj != 0:
        adjusted_total = float(forecast["yhat"].sum() * (1 + adj / 100))
        st.info(f"Adjusted total for horizon: **${adjusted_total:,.2f}**")

    # Download forecast CSV
    with open(os.path.join(FORECAST_DIR, f"forecast_{horizon}m_{int(ci*100)}ci.csv"), "rb") as f:
        st.download_button(
            label="📥 Download Forecast Data",
            data=f,
            file_name=f"forecast_{horizon}m_{int(ci*100)}ci.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Forecasting failed: {e}")

# ---------------- PDF Report Download ----------------
st.subheader("📄 Generate PDF Report")

# Ensure top_cats and top_merchants always exist
if "top_cats" not in locals():
    top_cats = pd.DataFrame(columns=["Category", "Total Spent"])
if "top_merchants" not in locals():
    top_merchants = pd.DataFrame(columns=["Description", "Total Spent"])

from report import generate_pdf_report

if st.button("📥 Create Full PDF Report"):
    # KPIs
    kpis = {
        "income": total_income,
        "expenses": total_expenses,
        "savings": net_savings,
        "rate": savings_rate,
    }

    # Save extra plots for report
    # Monthly income vs expenses
    monthly_plot = os.path.join(PLOTS_DIR, "monthly_plot.png")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    pivot_ie[["Income", "Expense"]].plot(ax=ax1, marker="o")
    ax1.set_title("Monthly Income vs Expenses")
    plt.tight_layout()
    fig1.savefig(monthly_plot, bbox_inches="tight")
    plt.close(fig1)

    # Savings rate trend
    savings_plot = os.path.join(PLOTS_DIR, "savings_plot.png")
    fig_sr, ax_sr = plt.subplots(figsize=(6, 3))
    sr.plot(ax=ax_sr, marker="o")
    ax_sr.set_title("Savings Rate Trend")
    plt.tight_layout()
    fig_sr.savefig(savings_plot, bbox_inches="tight")
    plt.close(fig_sr)

    # Pie chart
    pie_plot = os.path.join(PLOTS_DIR, "pie_plot.png")
    if not df_exp_only.empty:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        cat_totals.plot.pie(autopct="%1.1f%%", ax=ax_pie, ylabel="")
        ax_pie.set_title("Expense Distribution by Category")
        fig_pie.savefig(pie_plot, bbox_inches="tight")
        plt.close(fig_pie)

    # Top 5 tables (already computed)
    report_path = os.path.join(REPORT_DIR, "finance_report.pdf")

    generate_pdf_report(
        df_filt,
        kpis,
        cat_plot,
        time_plot,
        monthly_plot,
        savings_plot,
        pie_plot if not df_exp_only.empty else "",
        top_cats if not df_exp_only.empty else pd.DataFrame(),
        top_merchants if not df_exp_only.empty else pd.DataFrame(),
        ANOMALIES_FILE,
        os.path.join(FORECAST_DIR, f"forecast_{horizon}m_{int(ci*100)}ci.png"),
        report_path
    )

    with open(report_path, "rb") as f:
        st.download_button(
            label=" Download PDF Report",
            data=f,
            file_name="finance_report.pdf",
            mime="application/pdf"
        )

