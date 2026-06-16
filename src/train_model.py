import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# Load data
df = pd.read_csv('data/sales_data.csv')

print("✅ Data Loaded Successfully!")
print("Shape:", df.shape)
print(f"Average Sales: ₹{df['sales_amount'].mean():,.0f}\n")

# Features and Target
X = df.drop(['sales_amount', 'month'], axis=1)   # Remove month as it's just index
y = df['sales_amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/sales_model.pkl')

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("✅ Model Training Completed!")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: ₹{mae:,.0f}")

# Feature Importance
coefficients = model.coef_
features = X.columns

print("\nFeature Importance:")
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef:.2f}")