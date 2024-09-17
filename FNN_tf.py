import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'E:\MS_IITM\ME5201_Assignment\Training Dataset ME5201.xlsx'
# Load your dataset (replace 'your_dataset.xlsx' with your data file)
data = pd.read_excel(file_path)

# Define the independent variables (X) and the dependent variable (Y)
X = data[['Density of Fluid (kg/m3)', 'Bulk Modulus of Fluid (Pa)']]
Y = data['Sound wave speed in fluid (m/s)']

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Build a Feedforward Neural Network (FNN) with TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_val, Y_val), verbose=0)

# Predictions on validation set
Y_val_pred_scaled = model.predict(X_val).flatten()

# Inverse transform the scaled predictions to the original scale
Y_val_pred_original_scale = scaler.inverse_transform(np.vstack((np.zeros_like(Y_val_pred_scaled), Y_val_pred_scaled)).T)[:, 1]

# Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(Y_val, Y_val_pred_original_scale)
r2 = r2_score(Y_val, Y_val_pred_original_scale)

# Print the results
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

# Plot 3D scatter plot comparing actual vs predicted values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_val[:, 0], X_val[:, 1], Y_val, c='blue', marker='o', label='Actual Values')
ax.scatter(X_val[:, 0], X_val[:, 1], Y_val_pred_original_scale, c='red', marker='^', label='Predicted Values')
ax.set_xlabel('Density of Fluid (kg/m3)')
ax.set_ylabel('Bulk Modulus of Fluid (Pa)')
ax.set_zlabel('Sound wave speed in fluid (m/s)')
ax.set_title('3D Scatter Plot - Actual vs Predicted Values')
ax.legend()

plt.show()
