# 💰 Personal Finance AI

A Streamlit-based web app for analyzing personal finance transactions with **AI-powered insights**.  
Features include KPIs, visualizations, anomaly detection, expense forecasting, and PDF report generation.

---

## 🚀 Features
- Upload or use default transaction dataset (CSV).
- Clean and preprocess financial data.
- Interactive filters (by date and category).
- KPIs dashboard:
  - Total Income
  - Total Expenses
  - Net Savings
  - Savings Rate
- Visualizations:
  - Monthly Income vs Expenses
  - Savings Rate Trend
  - Expenses by Category (Bar)
  - Expenses Over Time (Line)
  - Expense Distribution (Pie)
- Top 5 Categories & Merchants by spend.
- Anomaly detection (Isolation Forest).
- Category prediction (ML classifier).
- Expense forecasting (Prophet).
- Generate **PDF Report** with all KPIs and visuals.

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend Models**: scikit-learn, Prophet
- **Data Handling**: pandas, matplotlib
- **Reports**: FPDF2
- **Storage**: CSV (future: database integration)

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AreebAshraf96/personal_finance_ai.git
   cd personal_finance_ai
   ```

2. Create and activate a virtual environment (Conda or venv recommended).

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Running the App

From the project root, run:
```bash
streamlit run src/dashboard.py
```

By default, Streamlit will launch at:  
👉 http://localhost:8501

---

## 📊 Example Workflow
1. Upload a CSV of transactions (Date, Description, Category, Amount).  
2. Explore KPIs, charts, and insights.  
3. Detect anomalies in spending.  
4. Forecast expenses for the next 3–12 months.  
5. Export a full **PDF Report**.  

---

## 🌐 Deployment
- Recommended: [Streamlit Cloud](https://streamlit.io/cloud)  
- Alternatives: Heroku, Render, Docker on AWS/GCP/Azure

---

## 🔮 Future Features
- User authentication & multi-user support.
- Database backend (SQLite/Postgres/MongoDB).
- Budget goals and alerts.
- Receipt OCR integration.
- Dark mode & UI polish.

---

## 📜 License
MIT License © 2025 Your Name

