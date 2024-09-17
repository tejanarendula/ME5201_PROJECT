#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
df = pd.read_excel("E:\MS_IITM\ME5201_Assignment\Training Dataset ME5201.xlsx")
df


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'df' is your DataFrame
# Replace 'x1', 'x2', and 'y' with the actual column names
x1_col = 'Density of Fluid (kg/m3)'
x2_col = 'Bulk Modulus of Fluid (Pa)'
y_col = 'Sound wave speed in fluid (m/s)'


# 3D Scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df[x1_col], df[x2_col], df[y_col], c='blue', marker='o')

# Set labels
ax.set_xlabel(x1_col)
ax.set_ylabel(x2_col)
ax.set_zlabel(y_col)

plt.title(f'3D Scatter Plot of {x1_col}, {x2_col} against {y_col}')
plt.show()


# In[38]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pandas as pd
df = pd.read_excel("E:\MS_IITM\ME5201_Assignment\Training Dataset ME5201.xlsx")
df

x = df['Density of Fluid (kg/m3)'].values
y = df['Bulk Modulus of Fluid (Pa)'].values
z = df['Sound wave speed in fluid (m/s)'].values




# # Reshape the data to have a single feature
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(-1, 1)

# Combine the features into a single array
features = np.concatenate((x, y), axis=1)

# Create polynomial features
poly_features = PolynomialFeatures(degree=2)
features_poly = poly_features.fit_transform(features)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(features_poly, z)


# Print the equation
equation = f"z = {model.intercept_[0]}"

for idx, coef in enumerate(model.coef_[0]):
    equation += f" + {coef}*x^({poly_features.powers_[idx][0]})*y^({poly_features.powers_[idx][1]})"

print("Equation:", equation)

# Predict the target variable
predicted_z = model.predict(features_poly)
print(model)
# Calculate evaluation metrics
mse = mean_squared_error(z, predicted_z)
rmse = np.sqrt(mse)
r2 = r2_score(z, predicted_z)

print("Mean Squared Error (MSE):     :", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2)                :", r2)
print()


# In[39]:


#x=488
#y=9.390000e+08
#z = 505.44776050240546 +  3.019294123509623e-08*x+ 7.547372118957641e-07*y + 0.00010848399670862397*x*x + -4.593776733405969e-10*x*y + 4.880628498030333e-17*y*y
#print(z)


# In[56]:


df = pd.read_excel("E:\MS_IITM\ME5201_Assignment\Training Dataset ME5201.xlsx")
df

x = df['Density of Fluid (kg/m3)'].values
y = df['Bulk Modulus of Fluid (Pa)'].values
z = df['Sound wave speed in fluid (m/s)'].values




# # Reshape the data to have a single feature
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(-1, 1)

# Create a meshgrid for 3D plot
x_range = np.linspace(x.min(), x.max(), 100)
y_range = np.linspace(y.min(), y.max(), 100)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)

# Combine the meshgrid features into a single array
meshgrid_features = np.c_[x_mesh.flatten(), y_mesh.flatten()]

# Create polynomial features for the meshgrid
meshgrid_poly = poly_features.transform(meshgrid_features)

# Predict the target variable for the meshgrid
z_mesh = model.predict(meshgrid_poly)
z_mesh = z_mesh.reshape(x_mesh.shape)

# ... (previous code)

# Scatter plot of the original data points
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='blue', label='Original Data Points')

# Plot the fitted surface
ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5, cmap='spring')

# Set axis labels
ax.set_xlabel('Density of Fluid (kg/m3)')
ax.set_ylabel('Bulk Modulus of Fluid (Pa)')
ax.set_zlabel('Sound wave speed in fluid (m/s)')

# Set the title
ax.set_title('Polynomial Regression Surface')

# Add a legend
ax.legend()


# Show the plot
plt.show()


# In[ ]:




