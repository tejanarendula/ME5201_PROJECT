import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting tools

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

# Calculate Mean Squared Errors
linear_test_mse = mean_squared_error(Y_test, Y_linear_test_pred)
linear_val_mse = mean_squared_error(Y_val, Y_linear_val_pred)

cubic_test_mse = mean_squared_error(Y_test, Y_cubic_test_pred)
cubic_val_mse = mean_squared_error(Y_val, Y_cubic_val_pred)

# Print the results
print("Linear Regression - Test Mean Squared Error:", linear_test_mse)
print("Linear Regression - Validation Mean Squared Error:", linear_val_mse)
print("\nCubic Regression - Test Mean Squared Error:", cubic_test_mse)
print("Cubic Regression - Validation Mean Squared Error:", cubic_val_mse)

# Retrieve coefficients for Linear Regression
linear_coefficients = np.append(linear_model.intercept_, linear_model.coef_)

# Retrieve coefficients for Cubic Regression
cubic_coefficients = np.append(cubic_model.intercept_, cubic_model.coef_)

# Print equations of fit
print("\nEquation of Linear Fit:")
print(f"Y = {linear_coefficients[0]:.4f} + {linear_coefficients[1]:.4f} * X1 + {linear_coefficients[2]:.4f} * X2")

print("\nEquation of Cubic Fit:")
print(f"Y = {cubic_coefficients[0]:.4f} + {cubic_coefficients[1]:.4f} * X1 + {cubic_coefficients[2]:.4f} * X2 + {cubic_coefficients[3]:.4f} * X1^2 + {cubic_coefficients[4]:.4f} * X1 * X2 + {cubic_coefficients[5]:.4f} * X2^2 + {cubic_coefficients[6]:.4f} * X1^3 + {cubic_coefficients[7]:.4f} * X1^2 * X2 + {cubic_coefficients[8]:.4f} * X1 * X2^2 + {cubic_coefficients[9]:.4f} * X2^3")


# Plot 3D scatter plot for Linear Regression
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_val, c='blue', marker='o', label='Actual Values (Validation Set)')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_linear_val_pred, c='red', marker='^', label='Linear Regression Predictions')
ax.set_xlabel('Density of Fluid (kg/m3)')
ax.set_ylabel('Bulk Modulus of Fluid (Pa)')
ax.set_zlabel('Sound wave speed in fluid (m/s)')
ax.set_title('3D Scatter Plot - Linear Regression vs Actual Values')
ax.legend()

# Plot 3D scatter plot for Cubic Regression
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_val, c='blue', marker='o', label='Actual Values (Validation Set)')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_cubic_val_pred, c='green', marker='^', label='Cubic Regression Predictions')
ax.set_xlabel('Density of Fluid (kg/m3)')
ax.set_ylabel('Bulk Modulus of Fluid (Pa)')
ax.set_zlabel('Sound wave speed in fluid (m/s)')
ax.set_title('3D Scatter Plot - Cubic Regression vs Actual Values')
ax.legend()

plt.show()

# Predict values for new input data
new_input_data = pd.DataFrame({'x1': [818], 'x2': [1.98e9]})
new_input_data_poly = poly.transform(new_input_data)
predicted_linear = linear_model.predict(new_input_data)
predicted_cubic = cubic_model.predict(new_input_data_poly)

print("\nPredicted Values for New Input Data:")
print(f"Linear Regression Prediction: {predicted_linear[0]:.4f}")
print(f"Cubic Regression Prediction: {predicted_cubic[0]:.4f}")
