import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = 'E:\MS_IITM\ME5201_Assignment\Training Dataset ME5201.xlsx'
# Load your dataset (replace 'your_dataset.xlsx' with your data file)
data = pd.read_excel(file_path)

# Define the independent variables (X) and the dependent variable (Y)
X = data[['Density of Fluid (kg/m3)', 'Bulk Modulus of Fluid (Pa)']]
Y = data['Sound wave speed in fluid (m/s)']

# Split the dataset into training, testing, and validation sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=0)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.33, random_state=0)  # 20% for testing, 10% for validation

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
Y_linear_test_pred = linear_model.predict(X_test)
Y_linear_val_pred = linear_model.predict(X_val)

# Cubic Regression
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
X_val_poly = poly.transform(X_val)

cubic_model = LinearRegression()
cubic_model.fit(X_train_poly, Y_train)
Y_cubic_test_pred = cubic_model.predict(X_test_poly)
Y_cubic_val_pred = cubic_model.predict(X_val_poly)

# Evaluate the models
linear_test_mse = mean_squared_error(Y_test, Y_linear_test_pred)
linear_val_mse = mean_squared_error(Y_val, Y_linear_val_pred)

cubic_test_mse = mean_squared_error(Y_test, Y_cubic_test_pred)
cubic_val_mse = mean_squared_error(Y_val, Y_cubic_val_pred)

print("Linear Regression - Test Mean Squared Error:", linear_test_mse)
print("Linear Regression - Validation Mean Squared Error:", linear_val_mse)

print("\nCubic Regression - Test Mean Squared Error:", cubic_test_mse)
print("Cubic Regression - Validation Mean Squared Error:", cubic_val_mse)

# Plot predictions vs actual values for comparison on the validation set
plt.scatter(Y_val, Y_linear_val_pred, label='Linear Regression', color='blue')
plt.scatter(Y_val, Y_cubic_val_pred, label='Cubic Regression', color='red')
plt.xlabel('Actual Values (Validation Set)')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Comparison of Linear and Cubic Regression on Validation Set')
plt.show()
