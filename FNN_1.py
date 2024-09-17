# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

file_path = 'E:\MS_IITM\ME5201_Assignment\Training Dataset ME5201.xlsx'
# Load your dataset (replace 'your_dataset.xlsx' with your data file)
data = pd.read_excel(file_path)

# Define the independent variables (X) and the dependent variable (Y)
X = data[['Density of Fluid (kg/m3)', 'Bulk Modulus of Fluid (Pa)']]
Y = data['Sound wave speed in fluid (m/s)']

# Split the dataset into training and testing sets (adjust test_size as needed)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a Feed Forward Neural Network (FNN) model
model = Sequential()
model.add(Dense(units=64, input_dim=2, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model using the training data
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Now you can use the trained FNN to make predictions on new data
