# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_csv("C:\\Users\\dell\\Downloads\\Self_project_non_core\\Sample - Superstore.csv", encoding='ISO-8859-1')

# Step 1.5: Initial Preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Step 1.6: Monthly Sales Summary and Trend Plotting
df['Month_Year'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Month_Year').agg({'Sales': 'sum'}).reset_index()
monthly_sales['Month_Year'] = monthly_sales['Month_Year'].astype(str)
monthly_sales['Month_Year'] = pd.to_datetime(monthly_sales['Month_Year'])

# Plot sales trend
plt.figure(figsize=(14,6))
plt.plot(monthly_sales['Month_Year'], monthly_sales['Sales'], marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature Engineering
monthly_sales['Month'] = monthly_sales['Month_Year'].dt.month
monthly_sales['Year'] = monthly_sales['Month_Year'].dt.year

X = monthly_sales[['Month', 'Year']]
y = monthly_sales['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model Training: Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Plot Predictions vs Actual
plt.figure(figsize=(14,6))
plt.plot(range(len(y_train)), y_train, label='Training Data', marker='o')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Actual Sales (Test)', marker='o')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_pred, label='Predicted Sales', marker='o')
plt.title('Sales Forecast vs Actual (Random Forest)')
plt.xlabel('Month Index')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R^2Score:", r2)