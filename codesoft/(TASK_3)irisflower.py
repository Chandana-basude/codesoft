import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the Iris dataset
df = sns.load_dataset('iris')

# Factorize the 'species' column
df['species'], _ = pd.factorize(df['species'])

# Display dataset information
print(df.head())
print("\nDataset description:")
print(df.describe())
print("\nMissing values in the dataset:")
print(df.isna().sum())

# 3D scatter plot - Petal dimensions
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.petal_length, df.petal_width, df.species, c=df.species, cmap='viridis')
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_zlabel('Species')
plt.title('3D Scatter Plot: Petal Dimensions')
plt.show()

# 3D scatter plot - Sepal dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.sepal_length, df.sepal_width, df.species, c=df.species, cmap='viridis')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
plt.title('3D Scatter Plot: Sepal Dimensions')
plt.show()

# 2D scatter plots
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", palette="viridis")
plt.title("Sepal Length vs. Sepal Width")
plt.show()

sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", palette="viridis")
plt.title("Petal Length vs. Petal Width")
plt.show()

# Elbow method for determining the optimal number of clusters
k_rng = range(1, 10)
sse = []

for k in k_rng:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(df[['petal_length', 'petal_width']])
    sse.append(km.inertia_)

plt.plot(k_rng, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

# KMeans clustering
km = KMeans(n_clusters=3, random_state=0)
y_predicted = km.fit_predict(df[['petal_length', 'petal_width']])
df['cluster'] = y_predicted

print("\nClustered Data:")
print(df.head())

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
true_labels = df['species']
predicted_labels = df['cluster']

cm = confusion_matrix(true_labels, predicted_labels)
class_labels = ['Setosa', 'Versicolor', 'Virginica']

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Fill matrix with values
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='white')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
