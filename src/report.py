import os
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# ---------------- PDF GENERATOR ----------------
def generate_pdf_report(
    df,
    kpis: dict,
    cat_plot: str,
    time_plot: str,
    monthly_plot: str,
    savings_plot: str,
    pie_plot: str,
    top_cats: pd.DataFrame,
    top_merchants: pd.DataFrame,
    anomalies_csv: str,
    forecast_plot: str,
    save_path: str
):
    pdf = FPDF()
    pdf.add_page()

    # ✅ Register fonts (regular + bold)
    pdf.add_font("DejaVu", "", r"C:\Users\hp\personal_finance_ai\fonts\dejavu-fonts-ttf-2.37\ttf\DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", r"C:\Users\hp\personal_finance_ai\fonts\dejavu-fonts-ttf-2.37\ttf\DejaVuSans-Bold.ttf")

    # Title
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 12, "Personal Finance AI - Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ---------------- KPIs ----------------
    pdf.set_font("DejaVu", "", 12)
    pdf.multi_cell(
        0, 10,
        f"Total Income: ${kpis['income']:,.2f}\n"
        f"Total Expenses: ${kpis['expenses']:,.2f}\n"
        f"Net Savings: ${kpis['savings']:,.2f}\n"
        f"Savings Rate: {kpis['rate']:.1f}%"
    )

    # ---------------- Charts ----------------
    def add_chart(title, path, width=120):
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.image(path, w=width)

    add_chart("Expenses by Category", cat_plot)
    add_chart("Expenses Over Time", time_plot)
    add_chart("Monthly Income vs Expenses", monthly_plot)
    add_chart("Savings Rate Trend", savings_plot)
    add_chart("Expense Distribution by Category", pie_plot)

    # ---------------- Top 5 Tables ----------------
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Top 5 Categories & Merchants", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, "Top 5 Categories:", new_x="LMARGIN", new_y="NEXT")
    for _, row in top_cats.iterrows():
        pdf.cell(0, 8, f" - {row['Category']}: {row['Total Spent']}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.cell(0, 10, "Top 5 Merchants:", new_x="LMARGIN", new_y="NEXT")
    for _, row in top_merchants.iterrows():
        pdf.cell(0, 8, f" - {row['Description']}: {row['Total Spent']}", new_x="LMARGIN", new_y="NEXT")

    # ---------------- Anomalies ----------------
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Anomaly Detection", new_x="LMARGIN", new_y="NEXT")
    anomalies_df = pd.read_csv(anomalies_csv)
    anomalies_count = len(anomalies_df[anomalies_df["Anomaly"] == "Anomaly"])
    pdf.set_font("DejaVu", "", 12)
    pdf.multi_cell(0, 10, f"Found {anomalies_count} anomalies in spending data.")

    # ---------------- Forecast ----------------
    add_chart("Expense Forecast", forecast_plot)

    pdf.output(save_path)
    print(f"✅ PDF Report generated at {save_path}")
