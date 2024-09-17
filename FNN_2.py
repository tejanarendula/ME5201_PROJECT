import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'E:\MS_IITM\ME5201_Assignment\Training Dataset ME5201.xlsx'
# Load your dataset (replace 'your_dataset.xlsx' with your data file)
data = pd.read_excel(file_path)

# Define the independent variables (X) and the dependent variable (Y)
X = data[['Density of Fluid (kg/m3)', 'Bulk Modulus of Fluid (Pa)']]
Y = data['Sound wave speed in fluid (m/s)']

# Split the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feedforward Neural Network (FNN)
fnn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0)
fnn_model.fit(X_train, Y_train)

# Predictions on validation set
Y_val_pred = fnn_model.predict(X_val)

# Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(Y_val, Y_val_pred)
r2 = r2_score(Y_val, Y_val_pred)

# Print the results
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

# Print the equation of the fit
coefficients = fnn_model.coefs_
intercepts = fnn_model.intercepts_

# Print the equation of the fit
equation = "Y = "
for i in range(len(coefficients)):
    equation += f"{coefficients[i]} * X{i+1} + "
equation += f"{intercepts[-1]}"

print("Equation of the fit:")
print(equation)

# Plot 3D scatter plot comparing actual vs predicted values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_val, c='blue', marker='o', label='Actual Values')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_val_pred, c='red', marker='^', label='Predicted Values')
ax.set_xlabel('Density of Fluid (kg/m3)')
ax.set_ylabel('Bulk Modulus of Fluid (Pa)')
ax.set_zlabel('Sound wave speed in fluid (m/s)')
ax.set_title('3D Scatter Plot - Actual vs Predicted Values')
ax.legend()

plt.show()
