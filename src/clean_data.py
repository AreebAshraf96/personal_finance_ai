import pandas as pd

def clean_transactions(file_path=r"C:\Users\hp\personal_finance_ai\data\synthetic_transactions.csv",
                       save_path=r"C:\Users\hp\personal_finance_ai\data\clean_transactions.csv"):
    """Clean and preprocess transaction data."""
    df = pd.read_csv(file_path)

    # Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop duplicates and missing
    df = df.drop_duplicates().dropna()

    # Normalize categories
    df["Category"] = df["Category"].str.title()

    # Add Income/Expense flag
    df["Type"] = df["Amount"].apply(lambda x: "Income" if x > 0 else "Expense")

    df.to_csv(save_path, index=False)
    print(f"✅ Cleaned dataset saved at {save_path} with {len(df)} rows")
    return df


# 👇 Add this to make script runnable directly
if __name__ == "__main__":
    df_clean = clean_transactions()
    print(df_clean.head())
