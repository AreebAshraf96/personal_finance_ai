import pandas as pd
import random
from datetime import datetime, timedelta

def create_synthetic_data(n=500, start="2024-01-01", end="2024-06-30",
                          save_path=r"C:\Users\hp\personal_finance_ai\data\synthetic_transactions.csv"):
    """Generate synthetic financial transactions and save to CSV."""
    categories = {
        "Food": ["McDonalds", "Subway", "Uber Eats", "KFC", "Dominos"],
        "Transport": ["Uber", "Lyft", "Shell", "BP Petrol", "Metro Ticket"],
        "Shopping": ["Amazon", "eBay", "Walmart", "Target", "Best Buy"],
        "Entertainment": ["Netflix", "Spotify", "Steam", "YouTube Premium"],
        "Bills": ["Electricity Co", "Water Works", "Internet ISP", "Phone Carrier"],
        "Income": ["Salary", "Freelance Payment", "Dividends"]
    }

    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    data = []
    for _ in range(n):
        category = random.choice(list(categories.keys()))
        merchant = random.choice(categories[category])
        date = start_date + timedelta(days=random.randint(0, (end_date-start_date).days))
        amount = round(random.uniform(500, 3000), 2) if category == "Income" else round(random.uniform(-200, -5), 2)
        data.append([date, merchant, category, amount])

    df = pd.DataFrame(data, columns=["Date", "Description", "Category", "Amount"])
    df = df.sort_values("Date").reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Synthetic dataset created at {save_path} with {len(df)} rows")
    return df

# 👇 Add this to make it runnable
if __name__ == "__main__":
    df = create_synthetic_data()
    print(df.head())
