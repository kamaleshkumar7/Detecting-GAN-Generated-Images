import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

# Load the feature vectors of GAN and real images and their corresponding labels
# /nfs/nfs2/home/rsantha/final_project_cv/method1/features_extraction/features_gan.npy
gan_features = np.load('features_gan_fn.npy')
real_features = np.load('features_real.npy')
gan_labels = np.zeros(len(gan_features))
real_labels = np.ones(len(real_features))
print(real_features)
print(gan_features)
X = np.concatenate((gan_features, real_features), axis=0)
y = np.concatenate((gan_labels, real_labels), axis=0)

# Split the data into train and test sets with 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for the SVM classifier
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
}

# Define the 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute the inverse class frequencies for balancing the losses of the two classes
class_weights = len(y_train) / (2 * np.bincount(y_train.astype(int)))

# Define the SVM classifier with RBF kernel and balanced sample weights
svm = SVC(kernel='rbf', class_weight='balanced')

# Define the grid search object with SVM classifier and parameter grid
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)




# Train the SVM classifier with feature vectors and corresponding labels
grid_search.fit(X_train, y_train, sample_weight=class_weights[y_train.astype(int)])

# Print the best hyperparameters and the corresponding accuracy
print("Best hyperparameters: ", grid_search.best_params_)
print("Accuracy: ", grid_search.best_score_)

dump(grid_search, 'svm_model.joblib')

# Predict the labels of test data using the fitted SVM classifier
y_pred = grid_search.predict(X_test)
print(y_pred)
print(y_test)
# Calculate the accuracy of predicted labels on test data
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy: ", accuracy)