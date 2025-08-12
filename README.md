# Task 6 — K-Nearest Neighbors (KNN) Classification

## Objective
Implement KNN for classification using the Iris dataset, tune K, and visualize results.

## Steps
1. Loaded and explored `Iris.csv`
2. Dropped unnecessary columns (`Id`)
3. Encoded the categorical `Species` column
4. Scaled features using StandardScaler
5. Split data into training and testing sets (80:20)
6. Trained KNN with K from 1 to 20
7. Found the best K (Optimal K = 2)
8. Evaluated with accuracy, confusion matrix, and classification report
9. Visualized:
   - Accuracy vs K (`accuracy_plot.png`)
   - Decision boundary with first 2 features (`decision_boundary.png`)

## Results
- **Optimal K:** 2
- **Accuracy:** 100%
- **Confusion Matrix:**
