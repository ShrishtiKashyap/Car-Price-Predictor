import pandas as pd
import numpy as np
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("vehicle-dataset-from-cardekho/car data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Feature Engineering
df["Car_Age"] = 2026 - df["Year"]

# Drop unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# Encode categorical features
df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["Seller_Type"] = df["Seller_Type"].map({
    "Dealer": 0,
    "Individual": 1
})

df["Transmission"] = df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

# Split features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nModel Performance")
print("----------------------")
print("R2 Score :", r2_score(y_test, predictions))
print("RMSE     :", np.sqrt(mean_squared_error(y_test, predictions)))

# Save model
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model saved successfully as car_price_model.pkl")

# Verify saved model
with open("car_price_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("✅ Model loaded successfully.")

print("\nSample Prediction:")
print(loaded_model.predict([X_test.iloc[0]]))