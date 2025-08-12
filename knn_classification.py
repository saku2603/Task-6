import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
df = pd.read_csv("Iris.csv")
print(df.head())

# Drop 'Id' if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Step 2: Encode target if needed
if df['Species'].dtype == 'object':
    df['Species'] = df['Species'].astype('category').cat.codes  # map to 0,1,2

# Features & Target
X = df.drop('Species', axis=1)
y = df['Species']

# Step 3: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 5: Find best K
accuracy_scores = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs K
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('K vs Accuracy')
plt.savefig("accuracy_plot.png")
plt.show()

# Step 6: Train with optimal K
optimal_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("Optimal K:", optimal_k)

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Step 7: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Decision boundary visualization (2 features for plotting)
X_plot = X_scaled[:, :2]
X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
    X_plot, y, test_size=0.2, random_state=42
)

knn_plot = KNeighborsClassifier(n_neighbors=optimal_k)
knn_plot.fit(X_train_plot, y_train_plot)

# Meshgrid
h = 0.02
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=df['Species'],
                palette='bright', edgecolor='k')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title('KNN Decision Boundary (First 2 Features)')
plt.savefig("decision_boundary.png")
plt.show()
