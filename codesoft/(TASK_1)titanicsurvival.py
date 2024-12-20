import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("C:/Users/basud/Downloads/Titanic-Dataset.csv")

# Display first 10 rows
print("First 10 rows of the dataset:")
print(df.head(10))

# Display shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# Summary statistics of the dataset
print("\nSummary statistics:")
print(df.describe())

# Count of survivals
if 'Survived' in df.columns:
    print("\nCount of Survivals:")
    print(df['Survived'].value_counts())
else:
    print("Column 'Survived' not found in the dataset")

# Visualize the count of survivals with respect to Pclass
if 'Pclass' in df.columns and 'Survived' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', hue='Pclass', data=df)
    plt.title("Survivals with respect to Pclass")
    plt.show()
else:
    print("Columns 'Pclass' or 'Survived' not found in the dataset")

# Visualize the count of survivals with respect to Gender
if 'Sex' in df.columns and 'Survived' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', hue='Survived', data=df)
    plt.title("Survivals with respect to Gender")
    plt.show()
else:
    print("Columns 'Sex' or 'Survived' not found in the dataset")

# Look at survival rate by sex
if 'Sex' in df.columns and 'Survived' in df.columns:
    print("\nSurvival rate by Sex:")
    print(df.groupby('Sex')[['Survived']].mean())

# Unique values in 'Sex' column
if 'Sex' in df.columns:
    print("\nUnique values in 'Sex':")
    print(df['Sex'].unique())

# Encoding the 'Sex' column
if 'Sex' in df.columns:
    labelencoder = LabelEncoder()
    df['Sex'] = labelencoder.fit_transform(df['Sex'])

# Display first few rows after encoding
if 'Sex' in df.columns and 'Survived' in df.columns:
    print("\nDataset after encoding 'Sex':")
    print(df[['Sex', 'Survived']].head())

# Visualize the count of survivals with respect to encoded Gender
if 'Sex' in df.columns and 'Survived' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', hue="Survived", data=df)
    plt.title("Survivals with respect to Encoded Gender")
    plt.show()

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isna().sum())

# Drop the 'Age' column if it exists
if 'Age' in df.columns:
    df = df.drop(['Age'], axis=1)

# Final dataset preview
df_final = df
print("\nFirst 10 rows of the final dataset:")
print(df_final.head(10))

# Splitting dataset into features and target variable
if 'Pclass' in df.columns and 'Sex' in df.columns and 'Survived' in df.columns:
    X = df[['Pclass', 'Sex']]
    Y = df['Survived']

    # Splitting into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Logistic Regression model
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    # Model prediction on the test set
    predictions = log.predict(X_test)
    print("\nModel predictions on the test set:")
    print(predictions)

    # Display true labels for the test set
    print("\nTrue labels for the test set:")
    print(Y_test.values)

    # Test the model with a new input
    res = log.predict([[2, 0]])

    # Display survival result
    if res == 0:
        print("\nSo sorry! Not Survived")
    else:
        print("\nSurvived")
else:
    print("Required columns ('Pclass', 'Sex', 'Survived') not found in the dataset.")
