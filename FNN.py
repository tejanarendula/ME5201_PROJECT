import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
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

# Feedforward Neural Network (FNN)
fnn_model = MLPRegressor(hidden_layer_sizes=(10, 1), max_iter=1000, random_state=0)
fnn_model.fit(X_train, Y_train)
Y_fnn_test_pred = fnn_model.predict(X_test)
Y_fnn_val_pred = fnn_model.predict(X_val)

# Calculate Mean Squared Errors and R2 Scores
fnn_test_mse = mean_squared_error(Y_test, Y_fnn_test_pred)
fnn_val_mse = mean_squared_error(Y_val, Y_fnn_val_pred)
fnn_test_r2 = r2_score(Y_test, Y_fnn_test_pred)
fnn_val_r2 = r2_score(Y_val, Y_fnn_val_pred)

# Print the results
print("FNN - Test Mean Squared Error:", fnn_test_mse)
print("FNN - Validation Mean Squared Error:", fnn_val_mse)
print("FNN - Test R2 Score:", fnn_test_r2)
print("FNN - Validation R2 Score:", fnn_val_r2)

# Plot 3D scatter plot for FNN
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_val, c='blue', marker='o', label='Actual Values (Validation Set)')
ax.scatter(X_val['Density of Fluid (kg/m3)'], X_val['Bulk Modulus of Fluid (Pa)'], Y_fnn_val_pred, c='purple', marker='^', label='FNN Predictions')
ax.set_xlabel('Density of Fluid (kg/m3)')
ax.set_ylabel('Bulk Modulus of Fluid (Pa)')
ax.set_zlabel('Sound wave speed in fluid (m/s)')
ax.set_title('3D Scatter Plot - FNN vs Actual Values')
ax.legend()

plt.show()
