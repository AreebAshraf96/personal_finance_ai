# =====================================================
# src/dashboard.py
# =====================================================

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests

from utils import plot_expenses_by_category, plot_expenses_over_time
from models import detect_anomalies, predict_category, forecast_expenses
from report import generate_pdf_report

# ----------------- BACKEND URL -----------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")


# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="Personal Finance AI", layout="wide")
st.title("🔐 Welcome to Personal Finance AI")

# =====================================================
# 🔐 LOGIN & REGISTER SECTION
# =====================================================
tab1, tab2 = st.tabs(["Login", "Register"])

# --- LOGIN TAB ---
with tab1:
    st.subheader("Login to your account")

    login_email = st.text_input("Email", key="login_email")
    login_password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", use_container_width=True, key="login_button"):
        if not login_email or not login_password:
            st.warning("Please enter both email and password.")
        else:
            try:
                response = requests.post(
                    f"{BACKEND_URL}/login",
                    json={"email": login_email, "password": login_password},
                )
                if response.status_code == 200:
                    st.session_state["user"] = login_email
                    st.success("✅ Login successful!")
                    st.rerun()  # refresh the app after login
                else:
                    st.error(response.json().get("error", "Login failed."))
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

# --- REGISTER TAB ---
with tab2:
    st.subheader("Create a new account")

    reg_email = st.text_input("Email", key="reg_email")
    reg_password = st.text_input("Password", type="password", key="reg_pass")

    if st.button("Register", use_container_width=True, key="register_button"):
        if not reg_email or not reg_password:
            st.warning("Please enter both email and password.")
        else:
            try:
                response = requests.post(
                    f"{BACKEND_URL}/register",
                    json={"email": reg_email, "password": reg_password},
                )
                if response.status_code == 201:
                    st.success("🎉 Registration successful! You can now log in.")
                else:
                    st.error(response.json().get("error", "Registration failed."))
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

# =====================================================
# 🧩 SESSION CHECK — SHOW DASHBOARD ONLY AFTER LOGIN
# =====================================================
if "user" not in st.session_state:
    st.info("👋 Please log in or register above to access your dashboard.")
    st.stop()

# =====================================================
# 🧍 USER INFO + LOGOUT BUTTON
# =====================================================
st.sidebar.success(f"Logged in as {st.session_state['user']}")
if st.sidebar.button("Logout", key="logout_button"):
    st.session_state.clear()
    st.rerun()

# =====================================================
# 📊 FETCH SAVED TRANSACTIONS (if any)
# =====================================================
df = pd.DataFrame()
try:
    resp = requests.post(
        f"{BACKEND_URL}/get_transactions",
        json={"email": st.session_state["user"]},
    )
    if resp.status_code == 200 and resp.json():
        df = pd.DataFrame(resp.json())
        st.sidebar.success("✅ Loaded saved transactions from MongoDB.")
    else:
        st.sidebar.info("No saved transactions found — upload new data.")
except Exception as e:
    st.sidebar.error(f"Could not fetch saved data: {e}")

# =====================================================
# 📂 ALLOW FILE UPLOAD / USE DEFAULT DATA
# =====================================================
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload your CSV",
    type=["csv"],
    key="file_upload_user_data"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# Make sure df exists
if df.empty:
    st.warning("⚠️ No data available. Please upload a CSV file.")
    st.stop()

# =====================================================
# ☁️ SAVE TRANSACTIONS TO MONGODB
# =====================================================
try:
    resp = requests.post(
        f"{BACKEND_URL}/save_transactions",
        json={
            "email": st.session_state["user"],
            "transactions": df.to_dict(orient="records"),
        },
    )
    if resp.status_code == 201:
        st.sidebar.success("✅ Transactions synced to MongoDB.")
    else:
        st.sidebar.warning("⚠️ Could not save to MongoDB.")
except Exception as e:
    st.sidebar.error(f"Database sync failed: {e}")

# =====================================================
# 🏠 MAIN DASHBOARD (simple preview)
# =====================================================
st.markdown("---")
st.subheader("📊 Preview of your data")
st.dataframe(df.head(20))
st.write("Your data has been securely saved in MongoDB per user account.")


# ---------------- THEME SWITCHER ----------------
def apply_theme(theme_choice: str):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    streamlit_dir = os.path.join(base_dir, ".streamlit")
    config_path = os.path.join(streamlit_dir, "config.toml")
    theme_file = os.path.join(streamlit_dir, f"config_{theme_choice.lower()}.toml")
    if os.path.exists(theme_file):
        shutil.copy(theme_file, config_path)
    else:
        st.warning(f"⚠️ Theme file '{theme_file}' not found. Using default.")

# Sidebar theme selector
theme_choice = st.sidebar.radio("🎨 Theme", ["Dark", "Light"], index=0)
apply_theme(theme_choice)


# ---------------- PATHS ----------------
BASE = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE, "data", "clean_transactions.csv")
PLOTS_DIR = os.path.join(BASE, "outputs", "plots")
ANOMALIES_FILE = os.path.join(BASE, "outputs", "anomalies", "anomalies.csv")
FORECAST_DIR = os.path.join(BASE, "outputs", "forecast")
REPORT_DIR = os.path.join(BASE, "outputs", "reports")
TMP_DIR = os.path.join(BASE, "outputs", "tmp")

for d in [PLOTS_DIR, FORECAST_DIR, TMP_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Overview", "📊 Spending Insights", "⚠️ Anomalies", "🔮 Forecast", "📄 Report"],
)

# ---------------- DATA HANDLING ----------------
@st.cache_data
def load_default_data(path):
    df_ = pd.read_csv(path)
    if "Date" not in df_.columns:
        for c in df_.columns:
            if "date" in c.lower():
                df_.rename(columns={c: "Date"}, inplace=True)
    df_["Date"] = pd.to_datetime(df_["Date"], errors="coerce").dt.tz_localize(None)
    return df_

uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Reading uploaded file..."):
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")
    st.sidebar.success("✅ File uploaded successfully!")
else:
    st.sidebar.info("Using default dataset.")
    df = load_default_data(DATA_FILE)

# ---------------- FILTERS ----------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
min_d, max_d = df["Date"].min(), df["Date"].max()

col_f1, col_f2 = st.sidebar.columns(2)
date_range = col_f1.date_input("📅 Date Range", (min_d.date(), max_d.date()))
categories = sorted(df["Category"].unique()) if "Category" in df.columns else []
selected_cats = col_f2.multiselect("🧾 Categories", categories, default=categories)

df_filt = df.copy()
if isinstance(date_range, tuple):
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filt = df_filt[(df_filt["Date"] >= start_d) & (df_filt["Date"] <= end_d)]
if selected_cats:
    df_filt = df_filt[df_filt["Category"].isin(selected_cats)]

# =====================================================
# 🏠 OVERVIEW PAGE
# =====================================================
if page == "🏠 Overview":
    st.markdown("### 👋 Welcome to Your Finance Dashboard")
    st.write(
        "Analyse, forecast, and track your financial activities intelligently with AI-driven insights."
    )
    st.markdown("---")

    # KPIs
    total_income = df_filt[df_filt["Amount"] > 0]["Amount"].sum()
    total_expenses = abs(df_filt[df_filt["Amount"] < 0]["Amount"].sum())
    net_savings = total_income - total_expenses
    rate = (net_savings / total_income * 100) if total_income else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Total Income", f"${total_income:,.2f}")
    col2.metric("💸 Total Expenses", f"${total_expenses:,.2f}")
    col3.metric("💰 Net Savings", f"${net_savings:,.2f}")
    col4.metric("📈 Savings Rate", f"{rate:.1f}%")

    st.markdown("---")

    # Monthly chart
    st.subheader("📅 Monthly Income vs Expenses")
    monthly = df_filt.groupby([df_filt["Date"].dt.to_period("M"), "Type"])["Amount"].sum().reset_index()
    monthly["Date"] = monthly["Date"].dt.to_timestamp()
    pivot_ie = monthly.pivot(index="Date", columns="Type", values="Amount").fillna(0)
    pivot_ie["Income"] = pivot_ie.get("Income", 0)
    pivot_ie["Expense"] = abs(pivot_ie.get("Expense", 0))
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    pivot_ie[["Income", "Expense"]].plot(ax=ax1, marker="o")
    ax1.set_title("Monthly Income vs Expenses")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

# =====================================================
# 📊 SPENDING INSIGHTS PAGE
# =====================================================
elif page == "📊 Spending Insights":
    st.subheader("🧠 Spending Insights")

    cat_plot = os.path.join(PLOTS_DIR, "cat_plot.png")
    time_plot = os.path.join(PLOTS_DIR, "time_plot.png")
    plot_expenses_by_category(df_filt, cat_plot)
    plot_expenses_over_time(df_filt, time_plot)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cat_plot, caption="Expenses by Category")
    with col2:
        st.image(time_plot, caption="Expenses Over Time")

    df_exp = df_filt[df_filt["Amount"] < 0]
    if not df_exp.empty:
        st.subheader("🏆 Top Categories and Merchants")
        top_cats = (
            df_exp.groupby("Category")["Amount"].sum().abs().nlargest(5).reset_index()
        )
        top_cats.columns = ["Category", "Amount"]
        st.dataframe(top_cats)

# =====================================================
# ⚠️ ANOMALIES PAGE
# =====================================================
elif page == "⚠️ Anomalies":
    st.subheader("⚠️ Anomaly Detection")
    tmp_csv = os.path.join(TMP_DIR, "filtered.csv")
    df_filt.to_csv(tmp_csv, index=False)

    with st.spinner("Detecting anomalies..."):
        df_anom = detect_anomalies(tmp_csv, ANOMALIES_FILE)

    if not df_anom.empty and "Anomaly" in df_anom.columns:
        anoms = df_anom[df_anom["Anomaly"] == "Anomaly"]
        if not anoms.empty:
            st.warning(f"⚠️ Found {len(anoms)} anomalies in your filtered data!")
            st.dataframe(anoms.head(20))
            with open(ANOMALIES_FILE, "rb") as f:
                st.download_button("📥 Download Anomaly Report", f, "anomalies.csv")
        else:
            st.success("✅ No anomalies detected.")
    else:
        st.info("No anomaly data generated yet.")

# =====================================================
# 🔮 FORECAST PAGE
# =====================================================
elif page == "🔮 Forecast":
    st.subheader("📈 Expense Forecast")

    horizon = st.selectbox("Forecast period (months)", [3, 6, 12], index=0)
    ci_choice = st.selectbox("Confidence interval", ["95%", "90%", "80%"], index=0)
    ci_map = {"95%": 0.95, "90%": 0.9, "80%": 0.8}
    ci = ci_map[ci_choice]

    tmp_forecast = os.path.join(TMP_DIR, "forecast.csv")
    df_filt.to_csv(tmp_forecast, index=False)

    with st.spinner("Generating forecast..."):
        forecast = forecast_expenses(
            tmp_forecast,
            periods=horizon,
            interval_width=ci,
            plot_path=os.path.join(FORECAST_DIR, f"forecast_{horizon}.png"),
            csv_path=os.path.join(FORECAST_DIR, f"forecast_{horizon}.csv"),
        )

    st.image(os.path.join(FORECAST_DIR, f"forecast_{horizon}.png"))
    st.success(f"✅ Forecast generated for the next {horizon} months (CI: {ci_choice})")

# =====================================================
# 📄 REPORT PAGE
# =====================================================
elif page == "📄 Report":
    st.subheader("📄 Generate Financial Report")

    if st.button("🧾 Create Full PDF Report"):
        with st.spinner("Generating PDF report..."):
            # --- KPIs ---
            kpis = {
                "income": df_filt[df_filt["Amount"] > 0]["Amount"].sum(),
                "expenses": abs(df_filt[df_filt["Amount"] < 0]["Amount"].sum()),
            }
            kpis["savings"] = kpis["income"] - kpis["expenses"]
            kpis["rate"] = (kpis["savings"] / kpis["income"] * 100) if kpis["income"] else 0

            # --- Default forecast horizon (use 3 months if not chosen) ---
            default_horizon = 3

            # --- Paths ---
            report_path = os.path.join(REPORT_DIR, "finance_report.pdf")
            cat_plot = os.path.join(PLOTS_DIR, "cat_plot.png")
            time_plot = os.path.join(PLOTS_DIR, "time_plot.png")
            monthly_plot = os.path.join(PLOTS_DIR, "monthly_plot.png")
            savings_plot = os.path.join(PLOTS_DIR, "savings_plot.png")
            pie_plot = os.path.join(PLOTS_DIR, "pie_plot.png")
            forecast_plot = os.path.join(FORECAST_DIR, f"forecast_{default_horizon}.png")
            anomalies_csv = ANOMALIES_FILE

            # --- Make sure latest charts exist ---
            from utils import plot_expenses_by_category, plot_expenses_over_time
            plot_expenses_by_category(df_filt, cat_plot)
            plot_expenses_over_time(df_filt, time_plot)

            # --- Top spenders ---
            import pandas as pd
            top_cats = (
                df_filt[df_filt["Amount"] < 0]
                .groupby("Category")["Amount"]
                .sum()
                .abs()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            top_cats["Total Spent"] = top_cats["Amount"].apply(lambda x: f"${x:,.2f}")
            top_cats = top_cats[["Category", "Total Spent"]]

            desc_col = next(
                (c for c in df_filt.columns if "desc" in c.lower() or "merchant" in c.lower()), None
            )
            if desc_col:
                top_merchants = (
                    df_filt[df_filt["Amount"] < 0]
                    .groupby(desc_col)["Amount"]
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
                top_merchants = pd.DataFrame()

            # --- Generate PDF ---
            generate_pdf_report(
                df_filt,
                kpis,
                cat_plot,
                time_plot,
                monthly_plot,
                savings_plot,
                pie_plot,
                top_cats,
                top_merchants,
                anomalies_csv,
                forecast_plot,
                report_path,
            )

        st.success("✅ PDF Report Generated Successfully!")
        with open(report_path, "rb") as f:
            st.download_button("📥 Download PDF Report", f, "finance_report.pdf")


