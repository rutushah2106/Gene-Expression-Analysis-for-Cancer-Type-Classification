# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense

# Step 2: Load and preprocess the data
# Replace this with the path to your actual gene expression dataset
data = pd.read_csv('gene_expression_data.csv')

# Assume the last column is the target (cancer type)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the gene expression values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
X_train_clustered = kmeans.fit_transform(X_train)
X_test_clustered = kmeans.transform(X_test)

# Step 4: Apply classification algorithms

# Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Neural Network (using TensorFlow/Keras)
nn_model = Sequential()
nn_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))  # Change to 'softmax' for multi-class classification
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Change to 'categorical_crossentropy' for multi-class classification
nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
_, nn_accuracy = nn_model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", nn_accuracy)

# Step 5: Evaluate the models
# SVM Evaluation
print("SVM Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Random Forest Evaluation
print("Random Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Neural Network Evaluation
print("Neural Network Evaluation:")
print("Accuracy:", nn_accuracy)
