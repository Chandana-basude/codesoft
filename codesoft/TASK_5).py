import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load dataset
data = pd.read_csv(r"C:/Users/basud/Downloads/creditcard.csv/creditcard.csv")

# Display basic information
print(data.shape)
print(data.columns)
print(data.info())

# Check for missing values
print("Missing values:", data.isnull().sum())

# Drop the 'Time' column
data = data.drop('Time', axis=1)

# Check for duplicates
print("Any duplicates:", data.duplicated().any())
data.drop_duplicates(keep='first', inplace=True)
print("Number of duplicates removed:", data.duplicated().sum())

# Summary statistics
print(data.describe())

# Class distribution
print("Class distribution:", data['Class'].value_counts())

# Visualize class distribution
plt.figure(figsize=(6, 5))
sns.countplot(x=data['Class'], hue=data['Class'])
plt.title('Class Distribution (0: No Fraud, 1: Fraud)')
plt.show()

# Prepare features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Logistic Regression using Statsmodels
X_with_const = sm.add_constant(X)  # Adding constant for the intercept
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit()
print(result.summary())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Fit Logistic Regression model (sklearn)
Lreg = LogisticRegression(random_state=22)
Lreg.fit(X_train, y_train)

# Predict using the trained model
y_predict = Lreg.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
print(f"True Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive: {tp}")

# Classification report
print(classification_report(y_test, y_predict))

# Additional metrics
sensitivity = tp / (tp + fn)  # Recall
specificity = tn / (tn + fp)
precision = tp / (tp + fp)

print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Precision: {precision:.2f}")

# Plotting Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt="d", cmap="Blues", 
            xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, Lreg.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
