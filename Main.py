import pandas as pd
from sklearn.linear_model import LinearRegression

#  Load the dataset
data = pd.read_csv("data.csv")

#  Separate features (X) and target (y)
X = data[["area", "bedrooms", "age"]]
y = data["price"]

#  Create and train the model
model = LinearRegression()
model.fit(X, y)

#  Make a prediction
predicted_price = model.predict([[3300, 3, 15]])

print(f"Predicted price for 3300 sqft, 3 bed, 15-year-old house: ${predicted_price[0]:,.2f}")
