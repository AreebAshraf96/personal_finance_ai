# src/dashboard.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import plot_expenses_by_category, plot_expenses_over_time
from src.models import detect_anomalies, predict_category, forecast_expenses

# ---------------- Paths ----------------
BASE = r"C:\Users\hp\personal_finance_ai"
DATA_FILE = os.path.join(BASE, "data", "clean_transactions.csv")
PLOTS_DIR = os.path.join(BASE, "outputs", "plots")
ANOMALIES_FILE = os.path.join(BASE, "outputs", "anomalies", "anomalies.csv")
FORECAST_DIR = os.path.join(BASE, "outputs", "forecast")
TMP_DIR = os.path.join(BASE, "outputs", "tmp")

for d in [PLOTS_DIR, FORECAST_DIR, TMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Personal Finance AI", layout="wide")
st.title("💰 Personal Finance AI Dashboard")

# ---------------- Data Loading (with cache) ----------------
@st.cache_data
def load_default_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)
    df_["Date"] = pd.to_datetime(df_["Date"], errors="coerce")
    if "Type" not in df_.columns:
        # safety: rebuild Type if missing
        df_["Type"] = df_["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")
    return df_

uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Type" not in df.columns:
        df["Type"] = df["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")
    st.success("✅ File uploaded successfully!")
else:
    st.info("Using default dataset...")
    df = load_default_data(DATA_FILE)

# ---------------- Filters ----------------
st.subheader("🔎 Filters")

min_d, max_d = df["Date"].min(), df["Date"].max()
col_f1, col_f2 = st.columns(2)
date_range = col_f1.date_input(
    "Date range",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date(),
)
categories = sorted(df["Category"].dropna().unique().tolist())
selected_cats = col_f2.multiselect("Categories", options=categories, default=categories)

# Apply filters
df_filt = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(str(date_range[0])), pd.to_datetime(str(date_range[1]))
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
    pivot_ie["Expense"] = pivot_ie.get("Expense", 0).abs()

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

# Pie chart of category distribution (expenses only)
df_exp_only = df_filt[df_filt["Amount"] < 0].copy()
if not df_exp_only.empty:
    cat_totals = df_exp_only.groupby("Category")["Amount"].sum().abs().sort_values(ascending=False)
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    ax_pie.pie(cat_totals.values, labels=cat_totals.index, autopct="%1.1f%%", startangle=140)
    ax_pie.axis("equal")
    ax_pie.set_title("Expense Distribution by Category")
    st.pyplot(fig_pie)

# Top 5 categories / merchants by spend
st.subheader("🏆 Top Spenders")
col_t1, col_t2 = st.columns(2)
if not df_exp_only.empty:
    top_cats = (
        df_exp_only.groupby("Category")["Amount"].sum().abs().sort_values(ascending=False).head(5).reset_index()
    )
    top_cats["Total Spent"] = top_cats["Amount"].apply(lambda x: f"${x:,.2f}")
    top_cats = top_cats[["Category", "Total Spent"]]

    top_merchants = (
        df_exp_only.groupby("Description")["Amount"].sum().abs().sort_values(ascending=False).head(5).reset_index()
    )
    top_merchants["Total Spent"] = top_merchants["Amount"].apply(lambda x: f"${x:,.2f}")
    top_merchants = top_merchants[["Description", "Total Spent"]]

    with col_t1:
        st.write("**Top 5 Categories**")
        st.dataframe(top_cats)
    with col_t2:
        st.write("**Top 5 Merchants / Descriptions**")
        st.dataframe(top_merchants)

    # Downloads
    st.download_button(
        "📥 Download Top Categories (CSV)",
        data=top_cats.to_csv(index=False).encode("utf-8"),
        file_name="top_categories.csv",
        mime="text/csv",
    )
    st.download_button(
        "📥 Download Top Merchants (CSV)",
        data=top_merchants.to_csv(index=False).encode("utf-8"),
        file_name="top_merchants.csv",
        mime="text/csv",
    )

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

from src.report import generate_pdf_report

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
