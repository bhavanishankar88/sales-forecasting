import pandas as pd
import numpy as np
import os

np.random.seed(42)
num_months = 1000

# Generate features
month = np.arange(1, num_months + 1)
marketing_spend = np.random.uniform(5000, 50000, num_months)
num_salespeople = np.random.randint(5, 25, num_months)
holiday_season = np.random.choice([0, 1], num_months, p=[0.7, 0.3])
economic_index = np.random.uniform(80, 120, num_months)
competition_level = np.random.choice([0, 1, 2], num_months)   # 0=Low, 1=Medium, 2=High

# Generate Sales (Target)
sales = (
    80000 +
    (marketing_spend * 1.8) +
    (num_salespeople * 4500) +
    (holiday_season * 25000) +
    (economic_index * 800) -
    (competition_level * 15000) +
    np.random.normal(0, 12000, num_months)
)

df = pd.DataFrame({
    'month': month,
    'marketing_spend': marketing_spend.round(0),
    'num_salespeople': num_salespeople,
    'holiday_season': holiday_season,
    'economic_index': economic_index.round(1),
    'competition_level': competition_level,
    'sales_amount': sales.round(0)
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/sales_data.csv', index=False)

print("✅ Sales Dataset Created!")
print(f"Total Months: {len(df)}")
print(f"Average Monthly Sales: ₹{df['sales_amount'].mean():,.0f}")
print(df.head())