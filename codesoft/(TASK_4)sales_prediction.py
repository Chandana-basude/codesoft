import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm 

# Load the dataset
adv = pd.read_csv(r"C:/Users/basud/Downloads/advertising.csv")

# Display dataset info
print("Dataset Info:")
print(adv.info())
print("\nShape of the dataset:", adv.shape)

# Check for missing values
print("\nMissing values in each column:")
print(adv.isnull().sum())

# Display basic statistics
print("\nSummary statistics:")
print(adv.describe())

# Pairplot for features vs. target
sns.pairplot(adv, x_vars=['TV', 'Radio', 'Newspaper'], y_vars=['Sales'], kind='scatter')
plt.show()

# Boxplots for each feature
fig, axs = plt.subplots(3, figsize=(10, 10))
sns.boxplot(data=adv['TV'], color='red', ax=axs[0])
sns.boxplot(data=adv['Newspaper'], color='green', ax=axs[1])
sns.boxplot(data=adv['Radio'], color='blue', ax=axs[2])
plt.tight_layout()
plt.show()

# Histograms for each feature
fig, axs = plt.subplots(3, figsize=(10, 10))
sns.histplot(data=adv['TV'], color='red', kde=True, ax=axs[0])
sns.histplot(data=adv['Newspaper'], color='green', kde=True, ax=axs[1])
sns.histplot(data=adv['Radio'], color='blue', kde=True, ax=axs[2])
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = adv[['TV', 'Radio', 'Newspaper']].corr()
sns.heatmap(corr, annot=True, cmap='plasma')
plt.title("Correlation Heatmap")
plt.show()

# Prepare data for regression
X = adv[['TV', 'Radio', 'Newspaper']]
y = adv['Sales']

# Statsmodels OLS regression
X_with_constant = sm.add_constant(X)
model = sm.OLS(y, X_with_constant).fit()
print("\nOLS Regression Summary:")
print(model.summary())

# Residuals
residuals = model.resid
sns.histplot(residuals, bins=70, kde=True)
plt.title("Residuals Distribution")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_predict = lr.predict(X_test)

# Display predictions
print("\nPredicted values for the test set:")
print(y_predict)

# Display actual values
print("\nActual values for the test set:")
print(y_test.values)

# Model performance
mse = mean_squared_error(y_test, y_predict)
print("\nMean Squared Error:", mse)
