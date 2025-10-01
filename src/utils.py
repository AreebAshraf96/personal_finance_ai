import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
BASE = r"C:\Users\hp\personal_finance_ai"
PLOTS_DIR = os.path.join(BASE, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------- Plotting Functions ----------------
def plot_expenses_by_category(df, save_path=None):
    """
    Plot total expenses grouped by category and save as PNG.
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "expenses_by_category.png")

    df_expenses = df[df["Amount"] < 0]  # only expenses
    if df_expenses.empty:
        print("⚠️ No expense data available for category plot.")
        return

    category_totals = df_expenses.groupby("Category")["Amount"].sum().abs()

    category_totals.plot(
        kind="bar",
        title="Expenses by Category",
        ylabel="Total Spent",
        xlabel="Category",
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Category expenses plot saved at {save_path}")


def plot_expenses_over_time(df, save_path=None):
    """
    Plot monthly expense trend and save as PNG.
    """
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "expenses_over_time.png")

    df_expenses = df[df["Amount"] < 0].copy()
    if df_expenses.empty:
        print("⚠️ No expense data available for time plot.")
        return

    # Ensure datetime
    df_expenses["Date"] = pd.to_datetime(df_expenses["Date"], errors="coerce")
    df_expenses = df_expenses.dropna(subset=["Date"])

    monthly = df_expenses.groupby(df_expenses["Date"].dt.to_period("M"))["Amount"].sum()
    if monthly.empty:
        print("⚠️ Not enough data to plot monthly expenses.")
        return

    monthly.plot(
        kind="line",
        marker="o",
        title="Expenses Over Time",
        ylabel="Total Spent",
        xlabel="Month",
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Monthly expenses plot saved at {save_path}")


# ---------------- Runnable ----------------
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(BASE, "data", "clean_transactions.csv"))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    print("✅ Loaded cleaned dataset. Generating plots...")
    plot_expenses_by_category(df)
    plot_expenses_over_time(df)
